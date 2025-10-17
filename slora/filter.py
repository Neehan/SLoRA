#!/usr/bin/env python3
import torch
from typing import List, Dict, Any
from slora.gate import HeadGradientGate
from torch.utils.data import DataLoader
import wandb


def filter_pass(model, dataset, config: Dict[str, Any], accelerator, logger, data_collator) -> List[int]:
    """
    Run filtering pass to get accepted sample indices.

    Returns:
        accepted_sample_indices: list of dataset sample indices that were accepted
    """
    device = accelerator.device
    model_config = accelerator.unwrap_model(model).config

    gate_params = {
        "d_hidden": model_config.hidden_size,
        "vocab_size": model_config.vocab_size,
        "m": config["slora"]["m"],
        "k": config["slora"]["k"],
        "target_accept_rate": config["slora"]["target_accept_rate"],
        "initial_threshold": config["slora"]["initial_threshold"],
        "controller_lr": config["slora"]["controller_lr"],
        "burn_in": config["slora"]["burn_in"],
        "seed": config["slora"]["seed"],
        "device": str(device),
        "reorth_every": config["slora"]["reorth_every"],
        "k_topk": config["slora"]["k_topk"],
        "random": config["slora"]["random"],
        "subspace_decay": config["slora"]["subspace_decay"],
    }

    gate = HeadGradientGate(**gate_params)
    dataset_filter = DatasetFilter(gate, accelerator)

    novelty_score_ema = 1.0
    last_accept = 1

    logger.info("Starting filtering pass...")
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["per_device_train_batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )
    dataloader = accelerator.prepare(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            labels = batch["labels"]

            novelty, accept = dataset_filter.filter_batch(batch_idx, hidden_states, logits, labels)

            novelty_score_ema = gate.ema_decay * novelty_score_ema + (1 - gate.ema_decay) * novelty
            last_accept = int(accept)

        if batch_idx % 10 == 0:
            logger.info(
                f"Filtered {batch_idx}/{len(dataloader)} batches, "
                f"acceptance_rate={gate.acceptance_rate():.3f}"
            )
            if accelerator.is_main_process:
                wandb.log({
                    "train/gate/novelty": novelty,
                    "train/gate/novelty_avg": novelty_score_ema,
                    "train/gate/novelty_energy_ema": gate.novelty_ema,
                    "train/gate/current_novelty_threshold": gate.current_novelty_threshold,
                    "train/gate/accept": last_accept,
                    "train/gate/acceptance_rate": gate.acceptance_rate(),
                }, step=batch_idx)

    accepted_batch_indices = dataset_filter.get_accepted_indices()
    logger.info(
        f"Filtering complete: {len(accepted_batch_indices)}/{len(dataloader)} batches accepted "
        f"({100.0 * len(accepted_batch_indices) / len(dataloader):.1f}%)"
    )

    batch_size = config["training"]["per_device_train_batch_size"]
    accepted_sample_indices = []
    for batch_idx in accepted_batch_indices:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        accepted_sample_indices.extend(range(start_idx, end_idx))

    return accepted_sample_indices


class DatasetFilter:
    """
    Single-pass filter over dataset using gate accept/reject.
    Records accepted indices for subsequent vanilla LoRA training.
    DDP-safe: syncs z across processes, syncs accept decision.
    """

    def __init__(self, gate: HeadGradientGate, accelerator):
        self.gate = gate
        self.accelerator = accelerator
        self.accepted_indices: List[int] = []
        self.global_step = 0

    def filter_batch(
        self,
        batch_idx: int,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, bool]:
        """
        Evaluate batch through gate and record if accepted.
        DDP-safe: syncs z embedding and accept decision across all processes.

        Returns:
            novelty: the novelty score for this batch
            accept: whether this batch was accepted
        """
        with torch.no_grad():
            z = self.gate.embed(hidden_states, logits, labels)

            if self.accelerator.num_processes > 1:
                torch.distributed.all_reduce(z, op=torch.distributed.ReduceOp.SUM)

            novelty = self.gate.novelty(z, self.global_step)
            accept = self.gate.accept(novelty, self.global_step)

            if self.accelerator.num_processes > 1:
                accept_tensor = torch.tensor(
                    [int(accept)], device=self.accelerator.device
                )
                torch.distributed.all_reduce(
                    accept_tensor, op=torch.distributed.ReduceOp.MIN
                )
                accept = bool(accept_tensor.item())

            count_increment = 1.0 / self.accelerator.num_processes

            if accept:
                self.accepted_indices.append(batch_idx)
                self.gate.update(z, count_increment)

            self.gate.step(self.global_step, accept)
            self.global_step += 1

        return novelty, accept

    def get_accepted_indices(self) -> List[int]:
        """Return list of accepted batch indices."""
        return self.accepted_indices
