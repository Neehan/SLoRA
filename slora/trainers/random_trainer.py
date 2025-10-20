import torch
from slora.trainers.base import TokenGatingTrainer


class RandomTokenTrainer(TokenGatingTrainer):
    """Random token selection baseline."""

    def __init__(self, topk_tokens: float, padding_label: int, *args, **kwargs):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens
        self.rng = torch.Generator()
        self.rng.manual_seed(self.args.seed)

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = logits.size(0)
        k = max(1, int(self.topk_tokens * N))

        # More efficient: directly create boolean mask via random selection
        indices = torch.randperm(N, generator=self.rng, device=logits.device)
        mask = indices < k

        return mask
