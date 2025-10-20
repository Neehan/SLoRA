import torch
import torch.nn.functional as F
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class FisherInfoTrainer(BaseTokenGatingTrainer):
    """Select tokens with highest Fisher information p(1-p)."""

    def __init__(self, topk_tokens: float, padding_label: int, *args, **kwargs):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens

    def compute_token_mask(self, hiddens, logits, labels):
        N = logits.size(0)
        k = max(1, int(self.topk_tokens * N))

        probs = F.softmax(logits, dim=-1)
        target_probs = probs[torch.arange(N), labels.clamp_min(0)]
        fisher = target_probs * (1 - target_probs)  # Fisher information per token

        # Normalize and sample
        sampling_probs = fisher / fisher.sum()
        indices = torch.multinomial(sampling_probs, k, replacement=False)

        mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
        mask[indices] = True
        return mask
