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

    @staticmethod
    def bernoulli_sample(scores: torch.Tensor, k: int):
        """
        Bernoulli sampling with importance weights (Horvitz-Thompson estimator).

        Args:
            scores: [N] non-negative scores
            k: expected number of tokens to select

        Returns:
            mask: [N] boolean mask
            importance_weights: [N] weights for unbiased estimation
        """
        N = scores.size(0)
        p = scores / scores.sum()
        q = torch.clamp(p * (k / N), max=1.0)

        u = torch.rand_like(q)
        mask = u < q

        importance_weights = 1.0 / (N * q)
        return mask, importance_weights

    def compute_token_mask(
        self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ):
        """
        Return boolean mask [N] indicating which tokens to backprop.

        Args:
            hiddens: [N, d_hidden]
            logits: [N, vocab_size]
            labels: [N]

        Returns:
            mask: [N] boolean mask, or (mask, weights) tuple where weights are importance sampling weights
        """
        return torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)

    def _forward_model(self, model, inputs, num_items_in_batch):
        model_inputs = dict(inputs)
        model_inputs.pop("labels", None)

        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            model_inputs["num_items_in_batch"] = num_items_in_batch

        model_inputs["return_dict"] = True
        model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return outputs

    def _extract_hiddens_logits(self, outputs, labels, attention_mask):
        if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
            raise ValueError("Model did not return hidden states required for token gating.")

        hiddens = outputs.hidden_states[-1]
        logits = outputs.logits

        labels = F.pad(labels, (0, 1), value=self.padding_label)
        labels = labels[:, 1:].contiguous()

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

        return valid_hiddens, valid_logits, valid_labels, logits_flat

    def _validate_token_mask(self, token_mask, valid_count, device):
        if token_mask.dim() != 1:
            raise ValueError(f"Token mask must be 1-D, got shape {tuple(token_mask.shape)}.")
        token_mask = token_mask.reshape(-1)
        if token_mask.size(0) != valid_count:
            raise ValueError(f"Token mask length {token_mask.size(0)} does not match valid token count {valid_count}.")
        if token_mask.dtype != torch.bool:
            if torch.is_floating_point(token_mask):
                raise TypeError("Token mask must be boolean or integer, not floating point.")
            token_mask = token_mask.to(dtype=torch.bool)
        if token_mask.device != device:
            token_mask = token_mask.to(device)
        return token_mask

    def _compute_gated_loss(self, valid_hiddens, valid_logits, valid_labels, logits_flat):
        valid_count = valid_logits.size(0)
        zero_loss = logits_flat.sum() * 0.0

        if valid_count == 0:
            return zero_loss, zero_loss.new_tensor(1.0)

        with torch.no_grad():
            mask_result = self.compute_token_mask(valid_hiddens, valid_logits, valid_labels)
            if isinstance(mask_result, tuple):
                token_mask, importance_weights = mask_result
            else:
                token_mask = mask_result
                importance_weights = None

        token_mask = self._validate_token_mask(token_mask, valid_count, valid_logits.device)
        valid_labels = valid_labels.long()

        selected_logits = valid_logits[token_mask]
        selected_labels = valid_labels[token_mask]
        selected_count = selected_logits.size(0)

        if selected_count == 0:
            return zero_loss, 1.0

        label_smoothing = getattr(self.args, "label_smoothing_factor", 0.0)
        per_token_loss = F.cross_entropy(
            selected_logits.float(),
            selected_labels,
            reduction="none",
            label_smoothing=label_smoothing,
        )

        if importance_weights is not None:
            selected_weights = importance_weights[token_mask]
            loss_sum = (per_token_loss * selected_weights).sum()
            denom = 1.0
        else:
            loss_sum = per_token_loss.sum()
            denom = selected_count

        return loss_sum, denom

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if (
            not self.model.training  # type: ignore
            or self.compute_loss_func is not None
            or "labels" not in inputs
        ):
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask")

        outputs = self._forward_model(model, inputs, num_items_in_batch)
        valid_hiddens, valid_logits, valid_labels, logits_flat = self._extract_hiddens_logits(
            outputs, labels, attention_mask
        )
        loss_sum, denom = self._compute_gated_loss(valid_hiddens, valid_logits, valid_labels, logits_flat)

        loss = loss_sum / max(denom, 1.0)

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss
