import torch
from transformers import Trainer
from typing import Optional, Dict, Any
from slora.gate import HeadGradientGate
from slora.optimizers import GatedOptimizer, GatedLRScheduler
import json
from pathlib import Path


class SLoRATrainer(Trainer):
    """
    HuggingFace Trainer with head-gradient proxy gating.

    Gates BEFORE backward pass using only forward quantities:
    - hidden_states (pre-head activations)
    - logits (classifier outputs)
    - labels (target tokens)

    Skips backward pass entirely on redundant batches.
    """

    def __init__(
        self,
        gate_config: Optional[Dict[str, Any]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gate: Optional[HeadGradientGate] = None
        self.gate_config = gate_config

        self.last_novelty = 0.0
        self.novelty_score_ema = 1.0  # EMA of novelty scores (ratio), not raw energy
        self.last_accept = 1

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """Wrap optimizer and scheduler with gating logic."""
        super().create_optimizer_and_scheduler(num_training_steps)
        if self.gate_config is not None:
            self.optimizer = GatedOptimizer(self.optimizer)  # type: ignore
            if self.lr_scheduler is not None:
                self.lr_scheduler = GatedLRScheduler(self.lr_scheduler)  # type: ignore

    def _initialize_gate(self) -> None:
        """Initialize gate after model is set up."""
        if self.gate is not None or self.gate_config is None:
            return

        device = next(self.model.parameters()).device  # type: ignore
        config = self.model.config  # type: ignore

        gate_params = {
            "d_hidden": config.hidden_size,  # type: ignore
            "vocab_size": config.vocab_size,  # type: ignore
            "m": self.gate_config["m"],
            "k": self.gate_config["k"],
            "target_accept_rate": self.gate_config["target_accept_rate"],
            "initial_threshold": self.gate_config["initial_threshold"],
            "controller_lr": self.gate_config["controller_lr"],
            "burn_in": self.gate_config["burn_in"],
            "seed": self.gate_config["seed"],
            "device": str(device),
            "reorth_every": self.gate_config["reorth_every"],
            "k_topk": self.gate_config["k_topk"],
            "random": self.gate_config["random"],
            "subspace_decay": self.gate_config["subspace_decay"],
        }

        self.gate = HeadGradientGate(**gate_params)

        self.log(  # type: ignore
            {
                "gate/d_hidden": config.hidden_size,  # type: ignore
                "gate/m": gate_params["m"],
                "gate/k": gate_params["k"],
                "gate/target_accept_rate": gate_params["target_accept_rate"],
                "gate/initial_threshold": gate_params["initial_threshold"],
                "gate/controller_lr": gate_params["controller_lr"],
                "gate/burn_in": gate_params["burn_in"],
            }
        )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Override training step to gate before backward.

        WARNING: This carefully reproduces the base Trainer's training_step behavior
        with gating logic inserted. Any changes must maintain equivalence to base trainer
        when gate is disabled. Key invariants:
        - Loss normalization by gradient_accumulation_steps (line 139)
        - Multi-GPU loss averaging for logging (line 141-142)
        - Proper detachment of returned loss (line 157)
        """
        if self.gate_config is not None and self.gate is None:
            self._initialize_gate()

        if self.gate is None:
            return super().training_step(model, inputs, num_items_in_batch)

        assert isinstance(self.optimizer, GatedOptimizer)
        assert self.lr_scheduler is None or isinstance(
            self.lr_scheduler, GatedLRScheduler
        )

        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]
        inputs["output_hidden_states"] = True  # type: ignore

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            hidden_states = outputs.hidden_states[-1]  # type: ignore
            logits = outputs.logits  # type: ignore

        del inputs

        with torch.no_grad():
            z = self.gate.embed(hidden_states, logits, labels)

            # sum across all gpus to get the direction of overall z
            if self.accelerator.num_processes > 1:
                torch.distributed.all_reduce(z, op=torch.distributed.ReduceOp.SUM)

            novelty = self.gate.novelty(z, self.state.global_step)
            accept = self.gate.accept(novelty, global_step=self.state.global_step)

            if self.accelerator.num_processes > 1:
                accept_tensor = torch.tensor(
                    [int(accept)], device=self.accelerator.device
                )
                torch.distributed.all_reduce(
                    accept_tensor, op=torch.distributed.ReduceOp.MIN
                )
                accept = bool(accept_tensor.item())

        self.last_novelty = novelty
        # Update novelty score EMA (use same decay as gate)
        if self.gate is not None:
            decay = self.gate.ema_decay
            self.novelty_score_ema = (
                decay * self.novelty_score_ema + (1 - decay) * novelty
            )
        self.last_accept = int(accept)

        count_increment = 1.0 / self.accelerator.num_processes

        if (
            not self.model_accepts_loss_kwargs or num_items_in_batch is None
        ) and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if accept:
            self.accelerator.backward(loss)
            self.gate.update(z, count_increment)
            self.optimizer.mark_accept()
            if self.lr_scheduler is not None:
                self.lr_scheduler.mark_accept()

        ret_loss = loss.detach()
        self.gate.step(self.state.global_step, accept)

        return ret_loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Add gate statistics to logs."""
        if self.gate is not None:
            logs["gate/novelty"] = self.last_novelty
            logs["gate/novelty_avg"] = self.novelty_score_ema
            logs["gate/novelty_energy_ema"] = self.gate.novelty_ema
            logs["gate/current_novelty_threshold"] = self.gate.current_novelty_threshold
            logs["gate/accept"] = self.last_accept
            logs["gate/acceptance_rate"] = self.gate.acceptance_rate()
        super().log(logs, start_time)

    def _save_checkpoint(self, model, trial, metrics=None):
        """Save gate metrics in checkpoint."""
        checkpoint_folder = super()._save_checkpoint(model, trial)

        if self.gate is not None and checkpoint_folder is not None:
            gate_metrics = {
                "acceptance_rate": self.gate.acceptance_rate(),
                "accepted_steps": self.gate.accepted_count,
                "total_steps": self.state.global_step,
                "rejected_steps": self.state.global_step - self.gate.accepted_count,
                "last_novelty": self.last_novelty,
                "novelty_score_avg": self.novelty_score_ema,
                "novelty_energy_ema": self.gate.novelty_ema,
                "last_accept": self.last_accept,
            }

            gate_file = Path(checkpoint_folder) / "gate_metrics.json"
            with open(gate_file, "w") as f:
                json.dump(gate_metrics, f, indent=2)

        return checkpoint_folder
