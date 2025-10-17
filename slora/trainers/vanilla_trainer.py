#!/usr/bin/env python3
import torch
from transformers import Trainer
from typing import Optional, Dict, Any


class VanillaTrainer(Trainer):
    """
    Vanilla HuggingFace Trainer for baseline LoRA training.
    No gating - just standard training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
