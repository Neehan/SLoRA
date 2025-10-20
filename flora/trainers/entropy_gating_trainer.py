import torch
import torch.nn.functional as F
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class EntropyGatingTrainer(BaseTokenGatingTrainer):
    """Sample tokens proportionally to entropy (stratified sampling)."""

    def __init__(self, topk_tokens: float, topk_logits: int, padding_label: int, *args, **kwargs):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens
        self.topk_logits = topk_logits

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        N = logits.size(0)
        k = max(1, int(self.topk_tokens * N))

        topk_logits, _ = logits.topk(self.topk_logits, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        entropy = -(topk_probs * torch.log(topk_probs + 1e-10)).sum(dim=-1)

        sampling_probs = entropy / entropy.sum()
        indices = torch.multinomial(sampling_probs, k, replacement=False)
        mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
        mask[indices] = True

        importance_weights = 1.0 / (sampling_probs * N)
        return mask, importance_weights
