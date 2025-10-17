import torch
from transformers import Trainer
from typing import Optional, Dict, Any
from slora.filter import filter_pass
from torch.utils.data import Subset
import json
from pathlib import Path


class SLoRATrainer(Trainer):
    """
    HuggingFace Trainer with optional dataset-level filtering.
    """

    def __init__(
        self,
        gate_config: Optional[Dict[str, Any]],
        *args,
        **kwargs,
    ):
        if gate_config is not None:
            original_train_dataset = kwargs["train_dataset"]

            from slora.utils.logging import setup_logging
            import wandb
            import os
            logger = setup_logging()

            if kwargs["args"].process_index == 0 and "wandb" in kwargs["args"].report_to:
                wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=kwargs["args"].run_name,
                    reinit=False,
                    resume="allow"
                )

            accepted_indices = filter_pass(
                kwargs["model"],
                original_train_dataset,
                gate_config,
                kwargs["args"].accelerator,
                logger,
                kwargs["data_collator"]
            )

            kwargs["train_dataset"] = Subset(original_train_dataset, accepted_indices)
            logger.info(f"Filtered train dataset: {len(accepted_indices)}/{len(original_train_dataset)} samples")

        super().__init__(*args, **kwargs)
        self.gate_config = gate_config

    def _save_checkpoint(self, model, trial, metrics=None):
        """Save checkpoint."""
        checkpoint_folder = super()._save_checkpoint(model, trial)
        return checkpoint_folder
