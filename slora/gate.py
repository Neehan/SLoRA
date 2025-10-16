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
        self.novelty_sq_ema = initial_threshold**2
        self.current_novelty_threshold = 0.5  # start at median
        self.acceptance_rate_ema = 1.0  # start as 1.0 for burn in period
        self.ema_decay = 0.95

        self.rng = torch.Generator(device=device).manual_seed(seed)

        self.tensor_sketch = TensorSketch(
            d1=d_hidden, d2=vocab_size, m=m, seed=seed, device=self.device
        )

        self.sketch = FrequentDirections(
            m, k, reorth_every=reorth_every, device=device, dtype=torch.float32
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse error vectors: e = softmax(logits) - one_hot(label).
        Keeps only k_topk largest logits + gold label to reduce noise.

        Returns:
            idx: (N, k_topk+1) selected vocab indices
            sel_errors: (N, k_topk+1) error values
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
        sel_probs = torch.softmax(sel_logits, dim=1)

        is_gold = idx == safe_labels.unsqueeze(1)
        sel_errors = sel_probs - is_gold.to(sel_probs.dtype)
        sel_errors = sel_errors * valid_mask.unsqueeze(1).float()

        return idx, sel_errors

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

        idx, sel_errors = self._compute_sparse_errors(
            logits_flat, labels_flat, valid_mask
        )

        z_tokens = self.tensor_sketch.sketch_batch(h_masked, idx, sel_errors)
        z = z_tokens.sum(dim=0)

        # don't normalize to handle multi gpu summing correctly
        return z

    def _compute_ema_zscore(self, x: float, t: int) -> float:
        """Compute bias-corrected EMA z-score using global_step t."""
        # Update EMAs
        self.novelty_ema = self.ema_decay * self.novelty_ema + (1 - self.ema_decay) * x
        self.novelty_sq_ema = (
            self.ema_decay * self.novelty_sq_ema + (1 - self.ema_decay) * x**2
        )

        # Bias correction (t+1 to avoid division by zero at step 0)
        bias_correction = 1 - self.ema_decay ** (t + 1)
        m_hat = self.novelty_ema / bias_correction
        v_hat = self.novelty_sq_ema / bias_correction

        # Z-score
        variance = v_hat - m_hat**2
        z_score = (x - m_hat) / ((variance + 1e-8) ** 0.5)
        return z_score

    def novelty(self, z: torch.Tensor, global_step: int) -> float:
        """Compute novelty score using bias-corrected EMA z-score."""
        z_norm_sq = (z @ z).item()
        if z_norm_sq < 1e-8:
            return 0.0

        W = self.sketch.get_basis()

        if W.shape[1] == 0:
            raw_novel_energy = z_norm_sq
        else:
            proj = W.T @ z
            redundant_energy = (proj @ proj).item()
            raw_novel_energy = max(0.0, z_norm_sq - redundant_energy)

        z_score = self._compute_ema_zscore(raw_novel_energy, global_step)
        novelty_score = torch.sigmoid(torch.tensor(z_score, device=self.device)).item()
        return novelty_score

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
        """Update streaming basis with normalized sketch."""
        self.sketch.update(z)
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
                torch.tensor(self.current_novelty_threshold), min=0.0, max=1.0
            ).item()
        elif global_step > self.burn_in:
            steps_since_burnin = global_step - self.burn_in
            progress_scale = 1.0  # / (1 + self.burn_in - steps_since_burnin)
            error = self.acceptance_rate_ema - self.target_accept_rate
            self.current_novelty_threshold += (
                self.controller_lr * error * progress_scale
            )
            self.current_novelty_threshold = torch.clamp(
                torch.tensor(self.current_novelty_threshold), min=0.0, max=1.0
            ).item()

    def acceptance_rate(self) -> float:
        """Compute overall acceptance rate."""
        return self.acceptance_rate_ema

    def reset(self) -> None:
        """Reset gate to initial state."""
        self.sketch.reset()
        self.accepted_count = 0
