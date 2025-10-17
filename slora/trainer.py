from transformers import Trainer
from typing import Optional, Dict, Any
from slora.filter import filter_pass
from torch.utils.data import Subset
from slora.utils.logging import setup_logging
import wandb
import os


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
            logger = setup_logging()

            if self.accelerator.is_main_process and "wandb" in self.args.report_to:  # type: ignore
                if not wandb.run:
                    wandb.init(
                        project=os.getenv("WANDB_PROJECT", "huggingface"),
                        name=self.args.run_name,
                        resume="allow",
                    )

            # filter dataset
            accepted_indices = filter_pass(
                self.model,
                original_train_dataset,
                gate_config,
                self.accelerator,
                logger,
                self.data_collator,
            )

            self.train_dataset = Subset(original_train_dataset, accepted_indices)
            logger.info(
                f"Filtered train dataset: {len(accepted_indices)}/{len(original_train_dataset)} samples"
            )

            # Adjust max_steps to match filtered dataset size (1 pass over filtered data)
            per_device_batch_size = self.args.per_device_train_batch_size
            gradient_accumulation_steps = self.args.gradient_accumulation_steps
            num_processes = self.accelerator.num_processes
            samples_per_step = per_device_batch_size * gradient_accumulation_steps * num_processes
            adjusted_max_steps = len(accepted_indices) // samples_per_step

            if adjusted_max_steps < self.args.max_steps:
                logger.info(
                    f"Adjusting max_steps from {self.args.max_steps} to {adjusted_max_steps} "
                    f"to match filtered dataset size (1 pass over data)"
                )
                self.args.max_steps = adjusted_max_steps

    def _save_checkpoint(self, model, trial):
        """Save checkpoint."""
        checkpoint_folder = super()._save_checkpoint(model, trial)
        return checkpoint_folder
