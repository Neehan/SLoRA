import torch
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer
from flora.sketch import TensorSketch


class FLoRATrainer(BaseTokenGatingTrainer):
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
        weight_clip: float,
        padding_label: int,
        *args,
        **kwargs,
    ):
        super().__init__(padding_label, *args, **kwargs)
        self.topk_tokens = topk_tokens
        self.weight_clip = weight_clip

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
    ):
        """
        Sample tokens proportionally to gradient magnitude approximation.

        Sketch each token's h ⊗ e, use ||sketch|| as proxy for gradient magnitude,
        then sample k% tokens with probability proportional to sketch norm.
        """
        sketches = self.sketcher.sketch_batch(hiddens, logits, labels)
        scores = sketches.norm(dim=1)

        k = max(1, int(self.topk_tokens * scores.size(0)))
        mask, weights = self.pps_sample(scores, k)
        weights = torch.clamp(weights, max=self.weight_clip)
        return mask, weights
