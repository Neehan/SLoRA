import torch
import torch.nn.functional as F
from typing import Optional
from flora.trainers.base_token_gating_trainer import BaseTokenGatingTrainer


class LossGatingTrainer(BaseTokenGatingTrainer):
    def __init__(self, *args, topk_tokens: float, weight_clip: float, topk_logits: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk_tokens = topk_tokens
        self.weight_clip = weight_clip
        self.topk_logits = topk_logits
        self.hook_handles = []
        self.lora_norm_accumulator = None
        self.cached_valid_mask = None
        self.w_u = None

    def _install_hooks(self):
        def forward_hook(module, input, output):
            if not self.model.training:
                return

            h = input[0].detach()
            h_norm = h.float().norm(dim=-1)

            if self.lora_norm_accumulator is None:
                self.lora_norm_accumulator = h_norm
            else:
                self.lora_norm_accumulator = self.lora_norm_accumulator + h_norm

        for name, module in self.model.named_modules():
            if "lora_A" in name and hasattr(module, "default"):
                handle = module.default.register_forward_hook(forward_hook)
                self.hook_handles.append(handle)

        if len(self.hook_handles) == 0:
            raise RuntimeError("Could not find any LoRA modules to hook")

    def _remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.lora_norm_accumulator = None
        self.cached_valid_mask = None

    def compute_token_mask(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        N, V = logits.size(0), logits.size(1)
        p = logits.float().softmax(dim=-1)

        eps = float(getattr(self.args, "label_smoothing_factor", 0.0))
        if eps > 0:
            e = p - ((1 - eps) * F.one_hot(labels, V).float() + eps / V)
        else:
            e = p - F.one_hot(labels, V).float()

        topk_vals, topk_idx = logits.topk(self.topk_logits, dim=-1)
        e_topk = e.gather(1, topk_idx)

        w_u = self.model.base_model.model.lm_head.weight.data
        w_u_topk = w_u[topk_idx]

        u = (e_topk.unsqueeze(1) @ w_u_topk.transpose(-2, -1)).squeeze(1).norm(dim=-1)

        if self.lora_norm_accumulator is None or self.cached_valid_mask is None:
            raise RuntimeError("LoRA activations not captured. Hook may have failed.")

        x_flat = self.lora_norm_accumulator.reshape(-1)
        x_valid = x_flat[self.cached_valid_mask]

        if x_valid.size(0) != N:
            raise RuntimeError(f"Accumulated norms size {x_valid.size(0)} doesn't match tokens {N}")

        scores = (u * x_valid).clamp_min(0.0)

        k = max(1, min(N, int(self.topk_tokens * N)))
        mask, weights = self.pps_sample(scores + 1e-12, k)
        weights = torch.clamp(weights, max=self.weight_clip)

        return mask, weights

    def _extract_hiddens_logits(self, outputs, labels, attention_mask):
        valid_hiddens, valid_logits, valid_labels, logits_flat = super()._extract_hiddens_logits(
            outputs, labels, attention_mask
        )

        if self.model.training:
            labels_padded = F.pad(labels, (0, 1), value=self.padding_label)
            labels_shifted = labels_padded[:, 1:].contiguous()
            labels_flat = labels_shifted.view(-1)

            if attention_mask is not None:
                attention_mask_flat = attention_mask.view(-1).bool()
            else:
                attention_mask_flat = torch.ones_like(labels_flat, dtype=torch.bool)

            self.cached_valid_mask = (labels_flat != self.padding_label) & attention_mask_flat

        return valid_hiddens, valid_logits, valid_labels, logits_flat

    def _compute_gated_loss(self, valid_hiddens, valid_logits, valid_labels, logits_flat):
        try:
            return super()._compute_gated_loss(valid_hiddens, valid_logits, valid_labels, logits_flat)
        finally:
            self.lora_norm_accumulator = None
            self.cached_valid_mask = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            if len(self.hook_handles) == 0:
                self._install_hooks()
            return super().training_step(model, inputs, num_items_in_batch)
        except Exception:
            self._remove_hooks()
            raise

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self._remove_hooks()
