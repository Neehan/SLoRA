import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Dict


class TokenGatingTrainer(Trainer):
    """
    Base trainer with token-level gating.

    Children override compute_token_mask() to implement selection criteria.
    Baseline returns all True (standard LoRA).
    """

    def __init__(self, padding_label: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_label = padding_label

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
        outputs = model(**model_inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        hiddens = outputs.last_hidden_state
        logits = outputs.logits

        # Align with next-token prediction: logits/hiddens position t predict labels at t+1
        hiddens = hiddens[:, :-1, :].contiguous()
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask[:, 1:].contiguous()

        vocab_size = logits.size(-1)

        hiddens_flat = hiddens.view(-1, hiddens.size(-1))
        logits_flat = logits.view(-1, vocab_size)
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
        zero_loss = logits_flat.new_zeros(())

        if valid_count == 0:
            loss = zero_loss
        else:
            with torch.no_grad():
                token_mask = self.compute_token_mask(valid_hiddens, valid_logits, valid_labels)
            token_mask = token_mask.to(device=valid_logits.device, dtype=torch.bool)
            if token_mask.numel() != valid_count:
                token_mask = token_mask[:valid_count]

            valid_labels = valid_labels.long()

            selected_logits = valid_logits[token_mask]
            selected_labels = valid_labels[token_mask]
            selected_count = selected_logits.size(0)

            if selected_count == 0:
                loss = zero_loss
            else:
                label_smoothing = getattr(self.args, "label_smoothing_factor", 0.0)
                ce_kwargs = {"reduction": "sum"}
                if label_smoothing:
                    ce_kwargs["label_smoothing"] = label_smoothing

                loss_sum = F.cross_entropy(selected_logits, selected_labels, **ce_kwargs)
                loss = loss_sum / valid_count

        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss = loss * (
                self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
            )

        if return_outputs:
            return loss, outputs
        return loss
