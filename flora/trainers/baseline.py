from transformers import Trainer
from typing import Dict, Optional
import torch


class BaselineTrainer(Trainer):
    """Standard HF Trainer with no custom loss computation."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """Override to request output_hidden_states to match BaseTokenGatingTrainer."""
        # Add output_hidden_states to match the other trainer
        inputs = dict(inputs)
        inputs["output_hidden_states"] = True

        # Call parent's compute_loss
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
