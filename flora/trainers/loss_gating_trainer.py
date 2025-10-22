import torch
import torch.nn.functional as F
from typing import Optional
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class LossGatingTrainer(BaseTokenGatingTrainer):
    def __init__(self, *args, topk_tokens: float, weight_clip: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk_tokens = topk_tokens
        self.weight_clip = weight_clip
        self.w_u = None
        self.w_o_norm_factors = None

    def _get_norm_factors(self):
        if self.w_o_norm_factors is not None and self.w_u is not None:
            return self.w_o_norm_factors, self.w_u

        print("=" * 80)
        print("DEBUG: Model structure")
        print("=" * 80)
        for name, module in self.model.named_modules():
            if "o_proj" in name or "lm_head" in name:
                print(f"Name: {name}")
                print(f"Type: {type(module)}")
                print(f"Dir: {[x for x in dir(module) if not x.startswith('_')][:20]}")
                print("-" * 80)

        layers_dict = {}
        for name, module in self.model.named_modules():
            if ".self_attn.o_proj" in name or ".self_attn.out_proj" in name:
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part.isdigit():
                        layer_idx = int(part)
                        layers_dict[layer_idx] = module.weight.data
                        break

        if not layers_dict:
            raise RuntimeError("Could not find attention output projections (o_proj/out_proj)")

        w_o_list = [layers_dict[i] for i in sorted(layers_dict.keys())]

        for name, module in self.model.named_modules():
            if name.endswith("lm_head"):
                self.w_u = module.weight.data
                break

        if self.w_u is None:
            raise RuntimeError("Could not find language model head (lm_head)")

        d = w_o_list[0].size(0)
        self.w_o_norm_factors = [
            w_o.norm(p='fro').item() / (d ** 0.5) for w_o in w_o_list
        ]

        return self.w_o_norm_factors, self.w_u

    def _compute_norm_proxy(self, e: torch.Tensor):
        """
        Compute norm-only recursive proxy.

        Instead of propagating full vectors, propagate norms:
        ||ũ^(L)|| = ||e @ W_U||
        ||ũ^(ℓ)|| ≈ ||ũ^(ℓ+1)|| × (||W_O^(ℓ)||_F / sqrt(d))

        Returns: sum of layer-wise norms [N]
        """
        norm_factors, w_u = self._get_norm_factors()

        u_tilde_norm = (e @ w_u).norm(dim=-1)
        norm_sum = u_tilde_norm.clone()

        for factor in reversed(norm_factors):
            u_tilde_norm = u_tilde_norm * factor
            norm_sum = norm_sum + u_tilde_norm

        return norm_sum

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        N, V = logits.size(0), logits.size(1)
        p = logits.float().softmax(dim=-1)

        eps = float(getattr(self.args, "label_smoothing_factor", 0.0))
        if eps > 0:
            e = p - ((1 - eps) * F.one_hot(labels, V).float() + eps / V)
        else:
            e = p - F.one_hot(labels, V).float()

        p_true = p.gather(1, labels.view(-1, 1)).squeeze(1)
        p_sumsq = (p * p).sum(dim=-1)

        if eps > 0:
            y_norm_sq = (1 - eps) ** 2 + (eps ** 2) / (V - 1)
            p_dot_y = (1 - eps) * p_true + (eps / (V - 1)) * (1 - p_true)
        else:
            y_norm_sq = 1.0
            p_dot_y = p_true

        u2 = p_sumsq + y_norm_sq - 2.0 * p_dot_y
        u = torch.sqrt(u2.clamp_min(0.0) + 1e-12)

        u_tilde_norm_sum = self._compute_norm_proxy(e)

        scores = (u * u_tilde_norm_sum).clamp_min(0.0)

        k = max(1, min(N, int(self.topk_tokens * N)))
        mask, weights = self.pps_sample(scores + 1e-12, k)
        weights = torch.clamp(weights, max=self.weight_clip)

        return mask, weights

