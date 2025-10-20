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

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            # Don't mutate inputs - create new dict with extra key
            outputs = model(**inputs, output_hidden_states=True)
            hiddens = outputs.hidden_states[-1]
            logits = outputs.logits
            labels = inputs["labels"]

            vocab_size = logits.size(-1)

            hiddens_flat = hiddens.view(-1, hiddens.size(-1))
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)

            valid_mask = labels_flat != self.padding_label
            valid_hiddens = hiddens_flat[valid_mask]
            valid_logits = logits_flat[valid_mask]
            valid_labels = labels_flat[valid_mask]

            if valid_hiddens.size(0) == 0:
                return torch.tensor(0.0, device=logits.device)

            token_mask = self.compute_token_mask(valid_hiddens, valid_logits, valid_labels)

            # Only compute loss for selected tokens to prevent gradient flow through unselected ones
            selected_logits = valid_logits[token_mask]
            selected_labels = valid_labels[token_mask]

            if selected_logits.size(0) == 0:
                return torch.tensor(0.0, device=logits.device)

            loss = F.cross_entropy(selected_logits, selected_labels)

        if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        return loss.detach()
