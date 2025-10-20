import torch
import torch.nn.functional as F
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class FisherInfoTrainer(BaseTokenGatingTrainer):
    """Select tokens with highest Fisher information p(1-p)."""

    def __init__(self, topk_tokens: float, topk_logits: int, padding_label: int, *args, **kwargs):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens
        self.topk_logits = topk_logits

    def compute_token_mask(self, hiddens, logits, labels):
        N = logits.size(0)
        k = max(1, int(self.topk_tokens * N))

        topk_vals, topk_indices = logits.topk(self.topk_logits, dim=-1)
        target_labels = labels.clamp_min(0).unsqueeze(1)
        target_logits = logits.gather(1, target_labels).squeeze(1)

        combined_logits = torch.cat([topk_vals, target_logits.unsqueeze(1)], dim=1)
        log_sum_exp = torch.logsumexp(combined_logits, dim=-1)
        target_probs = torch.exp(target_logits - log_sum_exp)
        fisher = target_probs * (1 - target_probs)

        return self.bernoulli_sample(fisher, k)
