#!/usr/bin/env python3
import torch
from typing import List, Dict, Any
from slora.gate import HeadGradientGate
from torch.utils.data import DataLoader
import wandb


def filter_pass(
    model, dataset, config: Dict[str, Any], accelerator, logger, data_collator
) -> List[int]:
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

    import time

    logger.info("Starting filtering pass...")
    model.eval()

    filter_start_time = time.time()

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["per_device_train_batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )
    dataloader = accelerator.prepare(dataloader)

    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    max_steps = config["training"].get("max_steps", -1)
    total_batches = len(dataloader)
    total_steps = total_batches // grad_accum_steps

    if max_steps > 0 and max_steps < total_steps:
        total_steps = max_steps
        total_batches = max_steps * grad_accum_steps

    burn_in = config["slora"]["burn_in"]

    if burn_in >= total_steps:
        logger.warning(
            f"burn_in={burn_in} >= total_steps={total_steps}. "
            f"Gate will never exit burn-in phase! Consider lowering burn_in or using more data."
        )

    past_key_values = None
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= total_batches:
                break
            outputs = model(**batch, output_hidden_states=True, use_cache=True, past_key_values=past_key_values)
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            labels = batch["labels"]
            past_key_values = outputs.past_key_values

            optimizer_step = batch_idx // grad_accum_steps
            novelty, accept = dataset_filter.filter_batch(
                batch_idx, hidden_states, logits, labels, optimizer_step
            )

            novelty_score_ema = (
                gate.ema_decay * novelty_score_ema + (1 - gate.ema_decay) * novelty
            )
            last_accept = int(accept)

            if batch_idx % grad_accum_steps == 0:
                optimizer_step = batch_idx // grad_accum_steps
                logging_steps = config["training"].get("logging_steps", 10)
                if optimizer_step % logging_steps == 0:
                    logger.info(
                        f"Filter step {optimizer_step}, "
                        f"acceptance_rate={gate.acceptance_rate():.3f}"
                    )
                    if accelerator.is_main_process:
                        wandb.log(
                            {
                                "filter/gate/novelty": novelty,
                                "filter/gate/novelty_avg": novelty_score_ema,
                                "filter/gate/novelty_energy_ema": gate.novelty_ema,
                                "filter/gate/current_novelty_threshold": gate.current_novelty_threshold,
                                "filter/gate/accept": last_accept,
                                "filter/gate/acceptance_rate": gate.acceptance_rate(),
                                "filter_step": optimizer_step,
                            },
                        )

    accepted_batch_indices = dataset_filter.get_accepted_indices()
    filter_time = time.time() - filter_start_time

    logger.info(
        f"Filtering complete in {filter_time:.1f}s: "
        f"{len(accepted_batch_indices)}/{total_batches} batches accepted "
        f"({100.0 * len(accepted_batch_indices) / total_batches:.1f}%)"
    )

    if accelerator.is_main_process:
        wandb.log({"filter/total_time_seconds": filter_time})

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
        optimizer_step: int,
    ) -> tuple[float, bool]:
        """
        Evaluate batch through gate and record if accepted.
        DDP-safe: syncs z embedding and accept decision across all processes.

        Returns:
            novelty: the novelty score for this batch
            accept: whether this batch was accepted
        """
        z = self.gate.embed(hidden_states, logits, labels)

        if self.accelerator.num_processes > 1:
            torch.distributed.all_reduce(z, op=torch.distributed.ReduceOp.SUM)

        novelty = self.gate.novelty(z, optimizer_step)
        accept = self.gate.accept(novelty, optimizer_step)

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

        self.gate.step(optimizer_step, accept)

        return novelty, accept

    def get_accepted_indices(self) -> List[int]:
        """Return list of accepted batch indices."""
        return self.accepted_indices
