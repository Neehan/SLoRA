import torch
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class RandomTokenTrainer(BaseTokenGatingTrainer):
    """Random token selection baseline."""

    def __init__(self, topk_tokens: float, padding_label: int, *args, **kwargs):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = logits.size(0)
        k = max(1, int(self.topk_tokens * N))

        indices = torch.randperm(N, device=logits.device)
        mask = indices < k

        return mask
