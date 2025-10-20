import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Dict


class BaseTokenGatingTrainer(Trainer):
    """
    Base trainer with token-level gating.

    Children override compute_token_mask() to implement selection criteria.
    Baseline returns all True (standard LoRA).
    """

    def __init__(self, padding_label: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_label = padding_label
        self.model_accepts_loss_kwargs = False

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Return boolean mask [N] indicating which tokens to backprop.

        Args:
            hiddens: [N, d_hidden]
            logits: [N, vocab_size]
            labels: [N]

        Returns:
            mask: [N] boolean mask
        """
        return torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if self.compute_loss_func is not None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if "labels" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask")

        model_inputs = dict(inputs)
        model_inputs.pop("labels", None)

        if self.model_accepts_loss_kwargs:
            extra_kwargs: Dict[str, torch.Tensor] = {}
            if num_items_in_batch is not None:
                extra_kwargs["num_items_in_batch"] = num_items_in_batch
            model_inputs.update(extra_kwargs)

        model_inputs["return_dict"] = True
        model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            raise ValueError("Model did not return hidden states required for token gating.")
        hiddens = hidden_states[-1]
        logits = outputs.logits

        # Align with next-token prediction: pad labels then shift (matching HF's ForCausalLMLoss)
        # Don't trim logits or hiddens - keep full sequence length
        import torch.nn.functional as F_pad
        labels = F_pad.pad(labels, (0, 1), value=self.padding_label)
        labels = labels[:, 1:].contiguous()
        # Note: attention_mask length matches original, labels now match too

        hiddens_flat = hiddens.view(-1, hiddens.size(-1))
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        if attention_mask is not None:
            attention_mask_flat = attention_mask.view(-1).bool()
        else:
            attention_mask_flat = torch.ones_like(labels_flat, dtype=torch.bool)

        valid_mask = (labels_flat != self.padding_label) & attention_mask_flat
        valid_hiddens = hiddens_flat[valid_mask]
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]

        valid_count = valid_logits.size(0)
        zero_loss = logits_flat.sum() * 0.0
        loss_sum = zero_loss
        denom = zero_loss.new_tensor(0.0)

        if valid_count != 0:
            with torch.no_grad():
                if self.model.training:
                    token_mask = self.compute_token_mask(valid_hiddens, valid_logits, valid_labels)
                else:
                    token_mask = torch.ones(valid_count, dtype=torch.bool, device=valid_logits.device)

            if token_mask.dim() != 1:
                raise ValueError(
                    f"Token mask must be 1-D, got shape {tuple(token_mask.shape)}."
                )
            token_mask = token_mask.reshape(-1)
            if token_mask.size(0) != valid_count:
                raise ValueError(
                    f"Token mask length {token_mask.size(0)} does not match valid token count {valid_count}."
                )
            if token_mask.dtype != torch.bool:
                if torch.is_floating_point(token_mask):
                    raise TypeError("Token mask must be boolean or integer, not floating point.")
                token_mask = token_mask.to(dtype=torch.bool)
            if token_mask.device != valid_logits.device:
                token_mask = token_mask.to(valid_logits.device)

            valid_labels = valid_labels.long()

            selected_logits = valid_logits[token_mask]
            selected_labels = valid_labels[token_mask]
            selected_count = selected_logits.size(0)

            if selected_count > 0:
                label_smoothing = getattr(self.args, "label_smoothing_factor", 0.0)
                # Upcast to float32 to match HF's ForCausalLMLoss behavior
                loss_sum = F.cross_entropy(
                    selected_logits.float(),
                    selected_labels,
                    reduction="sum",
                    label_smoothing=label_smoothing,
                )
                # Scale loss to match baseline magnitude
                # This ensures lr scheduling, logging, etc. work correctly
                scale_factor = valid_count / max(selected_count, 1)
                loss_sum = loss_sum * scale_factor
            else:
                loss_sum = zero_loss

            denom = valid_count

        loss = loss_sum / max(denom, 1.0)

        if (
            self.args.average_tokens_across_devices
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        if return_outputs:
            return loss, outputs
        return loss
