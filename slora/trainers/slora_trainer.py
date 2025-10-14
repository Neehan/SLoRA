import torch
from transformers import Trainer
from typing import Optional, Dict, Any
from slora.gate import SubspaceGate
from slora.hooks import lora_grad_vec, compute_lora_dim


class SLoRATrainer(Trainer):
    """
    HuggingFace Trainer subclass with Subspace-Gated Updates.

    Overrides training_step to:
    1. Compute loss and backward pass
    2. Extract LoRA gradients
    3. Gate optimizer step based on gradient novelty
    4. Update subspace sketch only on accepted steps
    """

    def __init__(
        self,
        gate_config: Optional[Dict[str, Any]] = None,
        enable_gate: bool = True,
        *args,
        **kwargs,
    ):
        """
        Args:
            gate_config: Dict with keys {m, k, tau_n, burn_in, seed, reorth_every}
            enable_gate: If False, behaves like standard Trainer (for baseline)
            *args, **kwargs: Passed to Trainer.__init__
        """
        super().__init__(*args, **kwargs)
        self.enable_gate = enable_gate
        self.gate = None
        self.gate_config = gate_config or {}
        self._last_accept = True
        self._cached_lora_params = []

        self.novelty_history = []
        self.accept_history = []

    def _initialize_gate(self) -> None:
        """Initialize gate after model is set up (called on first training step)."""
        if self.gate is not None:
            return

        assert self.model is not None, "Model not initialized"

        # Cache LoRA params once for consistency
        self._cached_lora_params = [
            (n, p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
            and ("lora_" in n.lower() or "lora_a" in n.lower() or "lora_b" in n.lower())
        ]

        d_lora = sum(p.numel() for _, p in self._cached_lora_params)
        device = next(self.model.parameters()).device

        gate_params = {
            "d_lora": d_lora,
            "m": self.gate_config["m"],
            "k": self.gate_config["k"],
            "tau_n": self.gate_config["tau_n"],
            "burn_in": self.gate_config["burn_in"],
            "seed": self.gate_config["seed"],
            "device": str(device),
            "reorth_every": self.gate_config["reorth_every"],
        }

        self.gate = SubspaceGate(**gate_params)

        self.log(
            {
                "gate/d_lora": d_lora,
                "gate/m": gate_params["m"],
                "gate/k": gate_params["k"],
                "gate/tau_n": gate_params["tau_n"],
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
        Override training step to only compute loss and backward (no stepping).
        Gating logic moved to optimizer_step.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss.detach()

    def _get_lora_grad_vec(self) -> torch.Tensor:
        """Extract flattened gradient vector from cached LoRA parameters."""
        grads = [
            p.grad.view(-1) for _, p in self._cached_lora_params if p.grad is not None
        ]
        assert len(grads) > 0, "No LoRA gradients found"
        return torch.cat(grads)

    def optimizer_step(self, epoch: int, **kwargs):
        """Override to gate optimizer updates based on gradient novelty."""
        if not self.enable_gate:
            return super().optimizer_step(epoch, **kwargs)

        if self.gate is None:
            self._initialize_gate()

        g_vec = self._get_lora_grad_vec()
        z = self.gate.embed(g_vec)
        novelty = self.gate.novelty(z)
        accept = self.gate.accept(novelty)

        self._last_accept = accept
        self.novelty_history.append(novelty)
        self.accept_history.append(int(accept))

        if accept:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )
            super().optimizer_step(epoch, **kwargs)
            self.gate.update(z)
        else:
            self.optimizer.zero_grad(set_to_none=True)

        self.gate.step()

        if self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "gate/novelty": novelty,
                    "gate/accept": int(accept),
                    "gate/acceptance_rate": self.gate.acceptance_rate(),
                    "gate/accepted_steps": self.gate.accepted_count,
                    "gate/total_steps": self.gate.step_count,
                }
            )

    def lr_scheduler_step(self, *args, **kwargs):
        """Keep scheduler in lockstep with optimizer."""
        if not self.enable_gate or self._last_accept:
            return super().lr_scheduler_step(*args, **kwargs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log to add gate statistics."""
        if self.enable_gate and self.gate is not None:
            logs["gate/acceptance_rate_overall"] = self.gate.acceptance_rate()
            logs["gate/accepted_steps_total"] = self.gate.accepted_count
            logs["gate/total_steps"] = self.gate.step_count
        super().log(logs, start_time)

    def _save_checkpoint(self, model, trial, metrics=None):
        """Override to save gate metrics in checkpoint."""
        checkpoint_folder = super()._save_checkpoint(model, trial, metrics)

        if self.enable_gate and self.gate is not None and checkpoint_folder is not None:
            import json
            from pathlib import Path

            gate_metrics = {
                "acceptance_rate": self.gate.acceptance_rate(),
                "accepted_steps": self.gate.accepted_count,
                "total_steps": self.gate.step_count,
                "rejected_steps": self.gate.step_count - self.gate.accepted_count,
                "novelty_history": self.novelty_history,
                "accept_history": self.accept_history,
            }

            gate_file = Path(checkpoint_folder) / "gate_metrics.json"
            with open(gate_file, "w") as f:
                json.dump(gate_metrics, f, indent=2)

        return checkpoint_folder
