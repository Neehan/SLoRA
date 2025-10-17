#!/usr/bin/env python3
import torch
from transformers import Trainer
from typing import Optional, Dict, Any, Tuple, Union
from torch import nn


class VanillaTrainer(Trainer):
    """
    Vanilla HuggingFace Trainer for baseline LoRA training.
    No gating - just standard training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute loss - ensures loss is always returned for eval."""
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard training step - no gating."""
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics."""
        super().log(logs, start_time)
