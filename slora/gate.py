import torch
from slora.directions import FrequentDirections
from slora.sketch import TensorSketch


class HeadGradientGate:
    """
    Head-gradient proxy gate: gates BEFORE backward pass using only forward quantities.

    Uses TensorSketch to compute sketch of head gradient ∇_W L = Σ_t h_t ⊗ e_t.
    Tracks low-rank subspace of accepted gradients via FrequentDirections.
    Gates based on directional novelty relative to tracked subspace.

    Cost: O(batch_size * seq_len * (d_hidden + vocab_size + m log m))
    """

    def __init__(
        self,
        d_hidden: int,
        vocab_size: int,
        m: int,
        k: int,
        target_accept_rate: float,
        controller_lr: float,
        burn_in: int,
        seed: int,
        device: str,
        reorth_every: int,
        k_topk: int,
        initial_threshold: float,
        random: bool,
        subspace_decay: float,
    ):
        assert k <= m

        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.m = m
        self.k = k
        self.target_accept_rate = target_accept_rate
        self.controller_lr = controller_lr
        self.burn_in = burn_in
        self.k_topk = k_topk
        self.device = torch.device(device)
        self.random = random

        self.novelty_ema = initial_threshold
        self.current_novelty_threshold = initial_threshold
        self.acceptance_rate_ema = 1.0  # start as 1.0 for burn in period
        self.ema_decay = 0.95

        self.rng = torch.Generator(device=device).manual_seed(seed)

        self.sketch = TensorSketch(
            d1=d_hidden, d2=vocab_size, m=m, seed=seed, device=self.device
        )

        self.subspace = FrequentDirections(
            m,
            k,
            reorth_every=reorth_every,
            device=device,
            dtype=torch.float32,
            decay=subspace_decay,
        )

        self.accepted_count = 0

    def _prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flatten and mask inputs for batch processing.

        Returns:
            h_masked: (N, d_hidden) hidden states with padding zeroed
            logits_flat: (N, vocab_size)
            labels_flat: (N,)
            valid_mask: (N,) boolean mask for non-padding tokens
        """
        B, T, H = hidden_states.shape
        N = B * T

        h_flat = hidden_states.reshape(N, H)
        logits_flat = logits.reshape(N, -1).detach()
        labels_flat = labels.reshape(N)

        valid_mask = labels_flat >= 0
        h_masked = h_flat * valid_mask.unsqueeze(1).float()

        return h_masked, logits_flat, labels_flat, valid_mask

    def _compute_sparse_errors(
        self,
        logits_flat: torch.Tensor,
        labels_flat: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sparse error vectors: e = softmax(logits) - one_hot(label).
        Keeps only k_topk largest logits + gold label + rest bucket to reduce noise.

        Returns:
            idx: (N, k_topk+1) selected vocab indices
            sel_errors: (N, k_topk+2) error values including rest bucket
            is_rest: (N, k_topk+2) boolean mask indicating rest bucket position
        """
        safe_labels = torch.where(
            valid_mask, labels_flat, torch.zeros_like(labels_flat)
        )

        topk_val, topk_idx = torch.topk(
            logits_flat, k=min(self.k_topk, logits_flat.size(1) - 1), dim=1
        )
        gold = safe_labels.unsqueeze(1)

        gold_in_topk = (topk_idx == gold).any(dim=1, keepdim=True)
        idx = torch.where(
            gold_in_topk.expand(-1, topk_idx.size(1) + 1),
            torch.cat([topk_idx, topk_idx[:, :1]], dim=1),
            torch.cat([topk_idx, gold], dim=1),
        )

        sel_logits = torch.gather(logits_flat, 1, idx)
        full_probs = torch.softmax(logits_flat, dim=1)
        sel_probs = torch.gather(full_probs, 1, idx)

        rest_prob = (1 - sel_probs.sum(dim=1, keepdim=True)).clamp(min=0.0)
        sel_probs_with_rest = torch.cat([sel_probs, rest_prob], dim=1)

        is_gold = idx == safe_labels.unsqueeze(1)
        is_gold_with_rest = torch.cat(
            [is_gold, torch.zeros_like(rest_prob, dtype=torch.bool)], dim=1
        )

        sel_errors = sel_probs_with_rest - is_gold_with_rest.to(
            sel_probs_with_rest.dtype
        )
        sel_errors = sel_errors * valid_mask.unsqueeze(1).float()

        idx_with_rest = torch.cat(
            [idx, torch.zeros_like(rest_prob, dtype=torch.long)], dim=1
        )
        is_rest = torch.cat(
            [torch.zeros_like(is_gold), torch.ones_like(rest_prob, dtype=torch.bool)],
            dim=1,
        )

        return idx_with_rest, sel_errors, is_rest

    def embed(
        self, hidden_states: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Sketch head gradient ∇_W L = Σ_t h_t ⊗ e_t via TensorSketch.

        Args:
            hidden_states: (B, T, d_hidden)
            logits: (B, T, vocab_size)
            labels: (B, T) with -100 for padding

        Returns:
            z ∈ R^m: Normalized sketch of aggregated head gradient
        """
        h_masked, logits_flat, labels_flat, valid_mask = self._prepare_inputs(
            hidden_states, logits, labels
        )

        idx, sel_errors, is_rest = self._compute_sparse_errors(
            logits_flat, labels_flat, valid_mask
        )

        non_rest_mask = ~is_rest
        idx_filtered = torch.where(non_rest_mask, idx, torch.zeros_like(idx))
        sel_errors_filtered = torch.where(
            non_rest_mask, sel_errors, torch.zeros_like(sel_errors)
        )

        z_tokens = self.sketch.sketch_batch(h_masked, idx_filtered, sel_errors_filtered)
        z = z_tokens[valid_mask].sum(dim=0)

        # don't normalize to handle multi gpu summing correctly
        return z

    def novelty(self, z: torch.Tensor, global_step: int) -> float:
        """
        Directional novelty via projection residual on a unit vector.
        Returns (raw_dir_novelty / EMA) so ~1.0 means 'as expected', >1.0 'more novel'.
        """
        z_norm = z.norm()
        if torch.isnan(z_norm) or z_norm.item() < 1e-12:
            return 0.0

        zn = z / z_norm  # unit direction

        W = self.subspace.get_basis()  # (m, r), assumed orthonormal columns
        if W.numel() == 0 or W.shape[1] == 0:
            raw_dir_novel = 1.0  # no basis yet: fully novel
        else:
            # residual energy in [0, 1]: 1 - ||W^T zn||^2
            coeff = W.T @ zn
            raw_dir_novel = max(0.0, 1.0 - float((coeff * coeff).sum()))

        # EMA for scale-free gating around 1.0 (keeps your existing thresholds)
        self.novelty_ema = (
            self.ema_decay * self.novelty_ema + (1 - self.ema_decay) * raw_dir_novel
        )
        return raw_dir_novel * z_norm.item() / (self.novelty_ema + 1e-8)

    def accept(self, novelty: float, global_step: int) -> bool:
        """
        Hard threshold acceptance based on novelty score.
        During burn-in, accept all updates.
        After burn-in, accept if novelty > threshold (never rejects high novelty).
        If random=True, ignore novelty and accept based on target rate only.
        """
        if global_step < self.burn_in:
            return True

        if self.random:
            return (
                torch.rand(1, generator=self.rng, device=self.device).item()
                < self.target_accept_rate
            )

        return novelty > self.current_novelty_threshold

    def update(self, z: torch.Tensor, count_increment: float) -> None:
        """Update streaming basis with sketch."""
        z_norm = z.norm()
        if torch.isnan(z_norm) or z_norm.item() < 1e-12:
            zn = z
        else:
            zn = z / z_norm

        self.subspace.update(zn)
        self.accepted_count += count_increment

    def step(self, global_step: int, accepted: bool) -> None:
        """Adapt threshold to maintain target acceptance rate using EMA."""
        self.acceptance_rate_ema = self.ema_decay * self.acceptance_rate_ema + (
            1 - self.ema_decay
        ) * float(accepted)

        if global_step > 2 * self.burn_in:
            error = self.acceptance_rate_ema - self.target_accept_rate
            self.current_novelty_threshold += self.controller_lr * error
            self.current_novelty_threshold = torch.clamp(
                torch.tensor(self.current_novelty_threshold), min=0.0
            ).item()
        elif global_step > self.burn_in:
            steps_since_burnin = global_step - self.burn_in
            progress_scale = 1.0  # / (1 + self.burn_in - steps_since_burnin)
            error = self.acceptance_rate_ema - self.target_accept_rate
            self.current_novelty_threshold += (
                self.controller_lr * error * progress_scale
            )
            self.current_novelty_threshold = torch.clamp(
                torch.tensor(self.current_novelty_threshold), min=0.0
            ).item()

    def acceptance_rate(self) -> float:
        """Compute overall acceptance rate."""
        return self.acceptance_rate_ema

    def reset(self) -> None:
        """Reset gate to initial state."""
        self.subspace.reset()
        self.accepted_count = 0
