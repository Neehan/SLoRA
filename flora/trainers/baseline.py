from transformers import Trainer
from typing import Dict, Optional
import torch
import torch.nn.functional as F


class BaselineTrainer(Trainer):
    """Standard HF Trainer with no custom loss computation."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """Override to manually compute loss like HF and print debug info."""
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0

        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Remove labels from inputs
        model_inputs = dict(inputs)
        model_inputs.pop("labels", None)
        model_inputs["output_hidden_states"] = True
        model_inputs["return_dict"] = True

        # Forward pass
        outputs = model(**model_inputs)
        logits = outputs.logits

        if self._debug_step < 2:
            print("\n" + "="*80)
            print(f"BASELINE TRAINER - Step {self._debug_step}")
            print("="*80)
            print(f"Model training: {model.training}")
            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"Labels shape (original): {labels.shape}")
            print(f"Logits shape: {logits.shape}")
            print(f"Logits dtype: {logits.dtype}")
            print(f"Labels[0,:10]: {labels[0, :10].tolist()}")
            print(f"Padding count: {(labels == -100).sum().item()}")

        # HF's ForCausalLMLoss approach
        # Line 55: Upcast to float
        logits_float = logits.float()

        # Lines 59-60: Pad and shift labels
        labels_padded = F.pad(labels, (0, 1), value=-100)
        shift_labels = labels_padded[..., 1:].contiguous()

        if self._debug_step < 2:
            print(f"After pad: {labels_padded.shape}")
            print(f"After shift: {shift_labels.shape}")
            print(f"Shift_labels[0,:10]: {shift_labels[0, :10].tolist()}")

        # Lines 63-64: Flatten
        logits_flat = logits_float.view(-1, logits_float.size(-1))
        labels_flat = shift_labels.view(-1)

        if self._debug_step < 2:
            print(f"Logits flat: {logits_flat.shape}")
            print(f"Labels flat: {labels_flat.shape}")
            print(f"Valid (non -100): {(labels_flat != -100).sum().item()}")

        # Line 67: fixed_cross_entropy with reduction='mean'
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction='mean')

        if self._debug_step < 2:
            print(f"Loss (mean): {loss.item():.10f}")
            print(f"First 3 logits[0]: {logits_flat[0, :3].tolist()}")
            self._debug_step += 1

        if return_outputs:
            return loss, outputs
        return loss
