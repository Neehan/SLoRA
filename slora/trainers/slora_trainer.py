import torch
from transformers import Trainer
from typing import Optional, Dict, Any
from slora.gate import HeadGradientGate
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

        self.novelty_history = []
        self.accept_history = []

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
        """
        if self.gate_config is not None and self.gate is None:
            self._initialize_gate()

        if self.gate is None:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]

        with self.compute_loss_context_manager():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            loss = outputs.loss

        with torch.no_grad():
            z = self.gate.embed(hidden_states, logits, labels)

            if self.accelerator.num_processes > 1:
                torch.distributed.all_reduce(z, op=torch.distributed.ReduceOp.AVG)
                z = z / (z.norm() + 1e-12)

            novelty = self.gate.novelty(z)
            accept = self.gate.accept(novelty, global_step=self.state.global_step)

            if self.accelerator.num_processes > 1:
                accept_tensor = torch.tensor(
                    [int(accept)], device=self.accelerator.device
                )
                torch.distributed.all_reduce(
                    accept_tensor, op=torch.distributed.ReduceOp.MIN
                )
                accept = bool(accept_tensor.item())

        self.novelty_history.append(novelty)
        self.accept_history.append(int(accept))

        count_increment = (
            1.0 / self.accelerator.num_processes
            if self.accelerator.num_processes > 1
            else 1.0
        )

        if accept:
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)
            self.gate.update(z, count_increment)
            ret_loss = loss.detach()
        else:
            ret_loss = loss.detach()

        self.gate.step(self.state.global_step, accept)

        return ret_loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Add gate statistics to logs."""
        if self.gate is not None:
            avg_novelty = (
                sum(self.novelty_history) / len(self.novelty_history)
                if self.novelty_history
                else 0.0
            )
            logs["gate/novelty"] = (
                self.novelty_history[-1] if self.novelty_history else 0.0
            )
            logs["gate/novelty_avg"] = avg_novelty
            logs["gate/current_novelty_threshold"] = self.gate.current_novelty_threshold
            logs["gate/accept"] = self.accept_history[-1] if self.accept_history else 1
            logs["gate/acceptance_rate"] = self.gate.acceptance_rate(
                self.state.global_step
            )
        super().log(logs, start_time)

    def _save_checkpoint(self, model, trial, metrics=None):
        """Save gate metrics in checkpoint."""
        checkpoint_folder = super()._save_checkpoint(model, trial)

        if self.gate is not None and checkpoint_folder is not None:
            avg_novelty = (
                sum(self.novelty_history) / len(self.novelty_history)
                if self.novelty_history
                else 0.0
            )
            gate_metrics = {
                "acceptance_rate": self.gate.acceptance_rate(self.state.global_step),
                "accepted_steps": self.gate.accepted_count,
                "total_steps": self.state.global_step,
                "rejected_steps": self.state.global_step - self.gate.accepted_count,
                "avg_novelty": avg_novelty,
                "novelty_history": self.novelty_history,
                "accept_history": self.accept_history,
            }

            gate_file = Path(checkpoint_folder) / "gate_metrics.json"
            with open(gate_file, "w") as f:
                json.dump(gate_metrics, f, indent=2)

        return checkpoint_folder
