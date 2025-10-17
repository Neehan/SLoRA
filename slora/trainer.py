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
        super().__init__(*args, **kwargs)
        self.gate_config = gate_config

        if gate_config is not None:
            original_train_dataset = self.train_dataset

            from slora.utils.logging import setup_logging
            import wandb
            import os
            logger = setup_logging()

            if self.accelerator.is_main_process and "wandb" in self.args.report_to:
                if not wandb.run:
                    wandb.init(
                        project=os.getenv("WANDB_PROJECT", "huggingface"),
                        name=self.args.run_name,
                        resume="allow"
                    )

            accepted_indices = filter_pass(
                self.model,
                original_train_dataset,
                gate_config,
                self.accelerator,
                logger,
                self.data_collator
            )

            self.train_dataset = Subset(original_train_dataset, accepted_indices)
            logger.info(f"Filtered train dataset: {len(accepted_indices)}/{len(original_train_dataset)} samples")

    def _save_checkpoint(self, model, trial, metrics=None):
        """Save checkpoint."""
        checkpoint_folder = super()._save_checkpoint(model, trial)
        return checkpoint_folder
