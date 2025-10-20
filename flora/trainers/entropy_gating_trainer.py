import torch
import torch.nn.functional as F
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class EntropyGatingTrainer(BaseTokenGatingTrainer):
    """Select tokens with highest prediction entropy (using top-k logits approximation)."""

    def __init__(self, topk_tokens: float, topk_logits: int, entropy_window: int, padding_label: int, *args, **kwargs):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens
        self.topk_logits = topk_logits
        self.entropy_window = entropy_window

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = logits.size(0)
        k = max(1, int(self.topk_tokens * N))

        topk_logits, _ = logits.topk(self.topk_logits, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        entropy = -(topk_probs * torch.log(topk_probs + 1e-10)).sum(dim=-1)

        if self.entropy_window > 1:
            window = self.entropy_window
            padding = window - 1
            entropy_padded = F.pad(entropy.unsqueeze(0), (padding, 0), value=0.0).squeeze(0)
            entropy = F.avg_pool1d(entropy_padded.unsqueeze(0).unsqueeze(0), kernel_size=window, stride=1).squeeze()

        _, topk_indices = entropy.topk(k)
        mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
        mask[topk_indices] = True

        return mask
