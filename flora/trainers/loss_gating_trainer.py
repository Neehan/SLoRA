import torch
import torch.nn.functional as F
from typing import Optional
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class LossGatingTrainer(BaseTokenGatingTrainer):
    def __init__(self, *args, topk_tokens: float, weight_clip: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk_tokens = topk_tokens
        self.weight_clip = weight_clip
        self.hook_handle = None
        self.cached_hiddens = None

    def _install_hook(self):
        target_module = None
        for name, module in self.model.named_modules():
            if "lora_A" in name and hasattr(module, "default"):
                target_module = module.default
                break

        if target_module is None:
            raise RuntimeError("Could not find LoRA module to hook")

        def forward_hook(module, input, output):
            self.cached_hiddens = input[0].detach()

        self.hook_handle = target_module.register_forward_hook(forward_hook)

    def _remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        self.cached_hiddens = None

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        N, V = logits.size(0), logits.size(1)
        p = logits.float().softmax(dim=-1)
        p_true = p.gather(1, labels.view(-1, 1)).squeeze(1)
        p_sumsq = (p * p).sum(dim=-1)

        eps = float(getattr(self.args, "label_smoothing_factor", 0.0))
        if eps > 0:
            y_norm_sq = (1 - eps) ** 2 + (eps ** 2) / (V - 1)
            p_dot_y = (1 - eps) * p_true + (eps / (V - 1)) * (1 - p_true)
        else:
            y_norm_sq = 1.0
            p_dot_y = p_true

        u2 = p_sumsq + y_norm_sq - 2.0 * p_dot_y
        u = torch.sqrt(u2.clamp_min(0.0) + 1e-12)

        if self.cached_hiddens is None:
            raise RuntimeError("LoRA activations not captured. Hook may have failed.")

        h_flat = self.cached_hiddens.reshape(-1, self.cached_hiddens.size(-1))
        if h_flat.size(0) != N:
            raise RuntimeError(f"Cached hiddens size {h_flat.size(0)} doesn't match tokens {N}")
        x = h_flat.float().norm(dim=-1)

        scores = (u * x).clamp_min(0.0)

        k = max(1, min(N, int(self.topk_tokens * N)))
        mask, weights = self.pps_sample(scores + 1e-12, k)
        weights = torch.clamp(weights, max=self.weight_clip)

        return mask, weights

    def _compute_gated_loss(self, valid_hiddens, valid_logits, valid_labels, logits_flat):
        try:
            return super()._compute_gated_loss(valid_hiddens, valid_logits, valid_labels, logits_flat)
        finally:
            self.cached_hiddens = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            if self.hook_handle is None:
                self._install_hook()
            return super().training_step(model, inputs, num_items_in_batch)
        except Exception:
            self._remove_hook()
            raise

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self._remove_hook()
