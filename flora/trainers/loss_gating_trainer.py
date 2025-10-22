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
        self.hook_handles = []
        self.h_norms_per_layer = None

    def _get_norm_factors(self):
        if self.w_o_norm_factors is not None and self.w_u is not None:
            return self.w_o_norm_factors, self.w_u

        layers_dict = {}
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn.o_proj") or name.endswith(".self_attn.out_proj"):
                parts = name.split(".")
                layer_idx = None
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break

                if layer_idx is not None:
                    if hasattr(module, 'base_layer'):
                        layers_dict[layer_idx] = module.base_layer.weight.data
                    else:
                        layers_dict[layer_idx] = module.weight.data

        if not layers_dict:
            raise RuntimeError("Could not find attention output projections (o_proj/out_proj)")

        w_o_list = [layers_dict[i] for i in sorted(layers_dict.keys())]

        for name, module in self.model.named_modules():
            if name.endswith("lm_head"):
                if hasattr(module, 'base_layer'):
                    self.w_u = module.base_layer.weight.data
                else:
                    self.w_u = module.weight.data
                break

        if self.w_u is None:
            raise RuntimeError("Could not find language model head (lm_head)")

        d = w_o_list[0].size(0)
        self.w_o_norm_factors = [
            w_o.norm(p='fro').item() / (d ** 0.5) for w_o in w_o_list
        ]

        return self.w_o_norm_factors, self.w_u

    def _install_hooks(self):
        norm_factors, _ = self._get_norm_factors()
        num_layers = len(norm_factors)

        self.h_norms_per_layer = [None] * num_layers

        def make_hook(layer_idx):
            def forward_hook(module, input, output):
                if not self.model.training:
                    return
                h = input[0].detach()
                h_norm = h.float().norm(dim=-1)
                self.h_norms_per_layer[layer_idx] = h_norm
            return forward_hook

        layer_idx = 0
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn.o_proj") or name.endswith(".self_attn.out_proj"):
                if hasattr(module, 'base_layer'):
                    target = module.base_layer
                else:
                    target = module
                handle = target.register_forward_hook(make_hook(layer_idx))
                self.hook_handles.append(handle)
                layer_idx += 1

        if len(self.hook_handles) == 0:
            raise RuntimeError("Could not find any attention output projections to hook")

    def _remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.h_norms_per_layer = None

    def _compute_layerwise_upstream_norms(self, e: torch.Tensor):
        """
        Compute ||ũ^(ℓ)|| for each layer via recursive propagation.

        ||ũ^(L)|| = ||e @ W_U||
        ||ũ^(ℓ)|| ≈ ||ũ^(ℓ+1)|| × (||W_O^(ℓ)||_F / sqrt(d))

        Returns: list of [||ũ^(0)||, ||ũ^(1)||, ..., ||ũ^(L)||] each [N]
        """
        norm_factors, w_u = self._get_norm_factors()

        u_tilde_norm = (e.to(w_u.dtype) @ w_u).norm(dim=-1)
        upstream_norms = [u_tilde_norm]

        for factor in reversed(norm_factors):
            u_tilde_norm = u_tilde_norm * factor
            upstream_norms.append(u_tilde_norm)

        upstream_norms.reverse()
        return upstream_norms

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

        upstream_norms = self._compute_layerwise_upstream_norms(e)

        if self.h_norms_per_layer is None:
            raise RuntimeError("Forward hooks did not capture activations. Hook installation may have failed.")

        score_sum = 0.0
        for layer_idx, (u_tilde_norm, h_norm) in enumerate(zip(upstream_norms, self.h_norms_per_layer)):
            if h_norm is None:
                raise RuntimeError(f"Layer {layer_idx} activation norm was not captured by hook")

            h_norm_flat = h_norm.view(-1)
            h_norm_valid = h_norm_flat[h_norm_flat.size(0) - N:]

            if h_norm_valid.size(0) != N:
                raise RuntimeError(f"Layer {layer_idx}: captured {h_norm_valid.size(0)} tokens, expected {N}")

            score_sum = score_sum + (u_tilde_norm * h_norm_valid)

        scores = (u * score_sum).clamp_min(0.0)

        k = max(1, min(N, int(self.topk_tokens * N)))
        mask, weights = self.pps_sample(scores + 1e-12, k)
        weights = torch.clamp(weights, max=self.weight_clip)

        return mask, weights

    def training_step(self, model, inputs, num_items_in_batch=None):
        if len(self.hook_handles) == 0:
            self._install_hooks()
        return super().training_step(model, inputs, num_items_in_batch)

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self._remove_hooks()
