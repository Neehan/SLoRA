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
        self._cached_lora_params = None

        self.novelty_history = []
        self.accept_history = []

    def _initialize_gate(self) -> None:
        """Initialize gate after model is set up (called on first training step)."""
        if self.gate is not None:
            return

        assert self.model is not None, "Model not initialized"

        # Cache LoRA params once for consistency
        self._cached_lora_params = [(n, p) for n, p in self.model.named_parameters()
                                    if p.requires_grad and ('lora_' in n.lower() or 'lora_a' in n.lower() or 'lora_b' in n.lower())]

        d_lora = sum(p.numel() for _, p in self._cached_lora_params)
        device = next(self.model.parameters()).device

        gate_params = {
            "d_lora": d_lora,
            "m": self.gate_config.get("m", 512),
            "k": self.gate_config.get("k", 64),
            "tau_n": self.gate_config.get("tau_n", 0.30),
            "burn_in": self.gate_config.get("burn_in", 2000),
            "seed": self.gate_config.get("seed", 0),
            "device": str(device),
            "reorth_every": self.gate_config.get("reorth_every", 128),
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
        Override training step to gate optimizer updates based on gradient novelty.
        """
        if not self.enable_gate:
            return super().training_step(model, inputs)

        if self.gate is None:
            self._initialize_gate()

        assert self.gate is not None, "Gate not initialized"
        assert self.optimizer is not None, "Optimizer not initialized"
        assert self.lr_scheduler is not None, "LR scheduler not initialized"

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        assert isinstance(loss, torch.Tensor), "Loss must be a tensor"

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            g_vec = lora_grad_vec(model)
            novelty = self.gate.novelty(g_vec)
            accept = self.gate.accept(g_vec)

            self.novelty_history.append(novelty)
            self.accept_history.append(int(accept))

            if accept:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.max_grad_norm
                )
                self.optimizer.step()  # type: ignore[union-attr]
                self.lr_scheduler.step()  # type: ignore[union-attr]
                self.gate.update(g_vec)

            self.optimizer.zero_grad()  # type: ignore[union-attr]
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

        return loss.detach()

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log to add gate statistics."""
        if self.enable_gate and self.gate is not None:
            logs["gate/acceptance_rate_overall"] = self.gate.acceptance_rate()
            logs["gate/accepted_steps_total"] = self.gate.accepted_count
        super().log(logs, start_time)
