import torch
from slora.trainers.base import TokenGatingTrainer
from slora.sketch import TensorSketch


class SLoRATrainer(TokenGatingTrainer):
    """
    Token-level gating using tensor sketch of head gradients.

    Sketches each token's h ⊗ e (head gradient approximation), then selects
    top-k% tokens by sketch norm for backpropagation.
    """

    def __init__(
        self,
        topk_tokens: float,
        sketch_dim: int,
        topk_logits: int,
        padding_label: int,
        *args,
        **kwargs,
    ):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens

        # Initialize sketcher
        device = next(self.model.parameters()).device  # type: ignore
        config = self.model.config  # type: ignore
        self.sketcher = TensorSketch(
            d_hidden=config.hidden_size,  # type: ignore
            vocab_size=config.vocab_size,  # type: ignore
            sketch_dim=sketch_dim,
            topk_logits=topk_logits,
            seed=self.args.seed,
            device=str(device),
        )

    def compute_token_mask(
        self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Select tokens by gradient magnitude approximation.

        Sketch each token's h ⊗ e, use ||sketch|| as proxy for gradient magnitude,
        then select top-k% tokens.
        """
        # Sketch each token's h ⊗ e (head gradient approximation)
        sketches = self.sketcher.sketch_batch(hiddens, logits, labels)

        # Score = ||sketch|| approximates gradient magnitude
        scores = sketches.norm(dim=1)

        # Select top-k% tokens by score
        k = max(1, int(self.topk_tokens * scores.size(0)))
        topk_indices = scores.topk(k).indices

        # Efficient scatter: use index comparison instead of zeros + assignment
        mask = torch.zeros(scores.size(0), dtype=torch.bool, device=logits.device)
        mask.scatter_(0, topk_indices, True)

        return mask
