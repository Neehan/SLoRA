#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import List, Dict, Any
from slora.gate import HeadGradientGate
from torch.utils.data import DataLoader
import wandb
import os


class FilterForward(nn.Module):
    """Compiled wrapper for filter pass forward that returns only tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.config.use_cache = False

    def forward(self, **inputs):
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1], outputs.logits


def filter_pass(
    model, dataset, gate_config: Dict[str, Any], accelerator, logger, data_collator
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
        "m": gate_config["m"],
        "k": gate_config["k"],
        "target_accept_rate": gate_config["target_accept_rate"],
        "initial_threshold": gate_config["initial_threshold"],
        "controller_lr": gate_config["controller_lr"],
        "burn_in": gate_config["burn_in"],
        "seed": gate_config["seed"],
        "device": str(device),
        "reorth_every": gate_config["reorth_every"],
        "k_topk": gate_config["k_topk"],
        "random": gate_config["random"],
        "subspace_decay": gate_config["subspace_decay"],
    }

    gate = HeadGradientGate(**gate_params)
    dataset_filter = DatasetFilter(gate, accelerator)

    novelty_score_ema = 1.0
    last_accept = 1

    import time

    logger.info("Starting filtering pass...")
    model.eval()

    # Compile forward wrapper for speedup (filter pass only, no gradients)
    # Extract base model without PEFT adapters (filter only needs pretrained weights)
    unwrapped_model = accelerator.unwrap_model(model)
    base_model = unwrapped_model.get_base_model()

    filter_forward = FilterForward(base_model).to(accelerator.device)
    filter_forward.eval()

    # Disable transformers' generic output collection wrapper for torch.compile
    os.environ["TRANSFORMERS_DISABLE_TELEMETRY"] = "1"
    torch._dynamo.config.suppress_errors = True

    filter_forward = torch.compile(
        filter_forward,
        backend="inductor",
        mode="max-autotune",
    )
    logger.info("Compiled filter forward pass")

    filter_start_time = time.time()

    dataloader = DataLoader(
        dataset,
        batch_size=gate_config["per_device_train_batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )
    dataloader = accelerator.prepare(dataloader)

    grad_accum_steps = gate_config["gradient_accumulation_steps"]
    max_steps = gate_config["max_steps"]
    total_batches = len(dataloader)
    total_steps = total_batches // grad_accum_steps

    if max_steps > 0 and max_steps < total_steps:
        total_steps = max_steps
        total_batches = max_steps * grad_accum_steps

    burn_in = gate_config["burn_in"]

    if burn_in >= total_steps:
        logger.warning(
            f"burn_in={burn_in} >= total_steps={total_steps}. "
            f"Gate will never exit burn-in phase! Consider lowering burn_in or using more data."
        )

    forward_time_total = 0.0
    gating_time_total = 0.0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= total_batches:
                break

            forward_start = time.time()
            forward_inputs = {
                k: v
                for k, v in batch.items()
                if k
                in ("input_ids", "attention_mask", "position_ids", "token_type_ids")
            }
            hidden_states, logits = filter_forward(**forward_inputs)
            labels = batch["labels"]
            forward_time_total += time.time() - forward_start

            optimizer_step = batch_idx // grad_accum_steps

            gating_start = time.time()
            novelty, accept = dataset_filter.filter_batch(
                batch_idx, hidden_states, logits, labels, optimizer_step
            )
            gating_time_total += time.time() - gating_start

            novelty_score_ema = (
                gate.ema_decay * novelty_score_ema + (1 - gate.ema_decay) * novelty
            )
            last_accept = int(accept)

            if batch_idx % grad_accum_steps == 0:
                optimizer_step = batch_idx // grad_accum_steps
                logging_steps = gate_config["logging_steps"]
                if optimizer_step % logging_steps == 0:
                    elapsed_time = time.time() - filter_start_time
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
                                "filter/elapsed_time_seconds": elapsed_time,
                                "filter_step": optimizer_step,
                            },
                        )

    accepted_batch_indices = dataset_filter.get_accepted_indices()
    filter_time = time.time() - filter_start_time

    forward_pct = 100.0 * forward_time_total / filter_time
    gating_pct = 100.0 * gating_time_total / filter_time

    logger.info(
        f"Filtering complete in {filter_time:.1f}s: "
        f"{len(accepted_batch_indices)}/{total_batches} batches accepted "
        f"({100.0 * len(accepted_batch_indices) / total_batches:.1f}%)"
    )
    logger.info(
        f"Time breakdown - Forward: {forward_time_total:.1f}s ({forward_pct:.1f}%), "
        f"Gating: {gating_time_total:.1f}s ({gating_pct:.1f}%)"
    )

    per_device_batch_size = gate_config["per_device_train_batch_size"]
    num_processes = accelerator.num_processes
    samples_per_batch = per_device_batch_size * num_processes

    accepted_sample_indices = []
    for batch_idx in accepted_batch_indices:
        start_idx = batch_idx * samples_per_batch
        end_idx = min(start_idx + samples_per_batch, len(dataset))
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
            accept_tensor = torch.tensor([int(accept)], device=self.accelerator.device)
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
