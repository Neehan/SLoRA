from transformers import Trainer
from typing import Dict, Optional
import torch


class BaselineTrainer(Trainer):
    """HF Trainer that uses MODEL's internal loss (true baseline) with debug prints."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """Use model's internal loss computation (HF default) with debug logging."""
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0

        # Make a copy to avoid modifying original
        inputs_copy = dict(inputs)
        inputs_copy["output_hidden_states"] = True

        # Call HF's default - this passes labels TO the model
        # Model computes loss internally via ForCausalLMLoss
        outputs = model(**inputs_copy)

        # Model returns loss when labels are provided
        loss = outputs.loss

        if self._debug_step < 2:
            print("\n" + "="*80)
            print(f"BASELINE TRAINER (Model Internal Loss) - Step {self._debug_step}")
            print("="*80)
            print(f"Model training: {model.training}")
            if 'input_ids' in inputs:
                print(f"Input shape: {inputs['input_ids'].shape}")
            if 'labels' in inputs:
                labels = inputs['labels']
                print(f"Labels shape: {labels.shape}")
                print(f"Labels[0,:10]: {labels[0, :10].tolist()}")
                print(f"Padding count: {(labels == -100).sum().item()}")
            print(f"Logits shape: {outputs.logits.shape}")
            print(f"Logits dtype: {outputs.logits.dtype}")
            print(f"Loss (from model.loss): {loss.item():.10f}")
            print(f"First 3 logits[0]: {outputs.logits[0, 0, :3].tolist()}")
            self._debug_step += 1

        if return_outputs:
            return loss, outputs
        return loss
