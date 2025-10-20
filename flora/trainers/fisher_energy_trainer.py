import torch
import torch.nn.functional as F
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class FisherEnergyTrainer(BaseTokenGatingTrainer):
    """Select tokens with highest Fisher energy ||p - y||²."""

    def __init__(self, topk_tokens: float, padding_label: int, *args, **kwargs):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = logits.size(0)
        k = max(1, int(self.topk_tokens * N))

        probs = F.softmax(logits, dim=-1)
        targets_onehot = F.one_hot(labels, num_classes=logits.size(-1)).float()
        fisher_energy = ((probs - targets_onehot) ** 2).sum(dim=-1)

        _, topk_indices = fisher_energy.topk(k)
        mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
        mask[topk_indices] = True

        return mask
