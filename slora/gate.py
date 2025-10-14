import torch
from slora.sketch import FrequentDirections


class CountSketchProjector:
    """
    Memory-efficient CountSketch projection (O(d) space, not O(d×m)).
    Uses feature hashing: z[h(i)] += s(i) * g[i] for i in [0, d).
    """

    def __init__(self, d: int, m: int, seed: int, device: torch.device):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.bucket = torch.randint(
            low=0, high=m, size=(d,), generator=g, dtype=torch.long
        ).to(device)
        self.sign = (
            (torch.randint(0, 2, (d,), generator=g, dtype=torch.int8) * 2 - 1).to(
                torch.int8
            )
        ).to(device)
        self.m = m
        self.device = device

    def project(self, g_vec: torch.Tensor) -> torch.Tensor:
        """Project g_vec: (d,) -> z: (m,) in fp32."""
        z = torch.zeros(self.m, dtype=torch.float32, device=g_vec.device)
        z.index_add_(0, self.bucket, self.sign.float() * g_vec.float())
        return z


class HeadGradientGate:
    """
    Head-gradient proxy gate: gates BEFORE backward pass using only forward quantities.

    Uses TensorSketch to compute sketch of head gradient ∇_W L = Σ_t h_t ⊗ e_t via:
    1. CountSketch h and e separately to m dimensions
    2. Convolve via FFT to get sketch of outer product

    Cost: O(batch_size * seq_len * (d_hidden + vocab_size + m log m))
    No explicit outer product formation.
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

        self.current_novelty_threshold = initial_threshold

        self.rng = torch.Generator(device=device).manual_seed(seed)

        g = torch.Generator(device="cpu").manual_seed(seed)
        self.bucket_h = torch.randint(
            0, m, (d_hidden,), generator=g, dtype=torch.long
        ).to(device)
        self.sign_h = (
            (
                torch.randint(0, 2, (d_hidden,), generator=g, dtype=torch.int8) * 2 - 1
            ).to(torch.int8)
        ).to(device)

        g_e = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.bucket_e = torch.randint(
            0, m, (vocab_size,), generator=g_e, dtype=torch.long
        ).to(device)
        self.sign_e = (
            (
                torch.randint(0, 2, (vocab_size,), generator=g_e, dtype=torch.int8) * 2
                - 1
            ).to(torch.int8)
        ).to(device)

        self.sketch = FrequentDirections(
            m, k, reorth_every=reorth_every, device=device, dtype=torch.float32
        )

        self.accepted_count = 0

    def embed(
        self, hidden_states: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Sketch head gradient via TensorSketch with sparse top-K error approximation.

        Args:
            hidden_states: (B, T, d_hidden)
            logits: (B, T, vocab_size)
            labels: (B, T)

        Returns:
            Normalized sketch ẑ ∈ R^m
        """
        B, T, H = hidden_states.shape
        N = B * T

        h_flat = hidden_states.reshape(N, H)
        logits_flat = logits.reshape(N, -1).detach()
        labels_flat = labels.reshape(N)

        # Handle padding tokens: HF uses -100 as ignore index
        # Replace -100 with 0 for safe indexing, then mask out contribution
        valid_mask = labels_flat >= 0
        safe_labels = torch.where(
            valid_mask, labels_flat, torch.zeros_like(labels_flat)
        )

        # Mask hidden states for padding tokens
        h_masked = h_flat * valid_mask.unsqueeze(1).float()

        s_h = torch.zeros(N, self.m, dtype=torch.float32, device=h_flat.device)
        s_h.index_add_(
            1, self.bucket_h, (self.sign_h.float().unsqueeze(0) * h_masked.float())
        )

        topk_val, topk_idx = torch.topk(
            logits_flat, k=min(self.k_topk, logits_flat.size(1) - 1), dim=1
        )
        gold = safe_labels.unsqueeze(1)
        idx = torch.cat([topk_idx, gold], dim=1)

        sel_logits = torch.gather(logits_flat, 1, idx)
        sel_probs = torch.softmax(sel_logits, dim=1)

        is_gold = idx == safe_labels.unsqueeze(1)
        sel_errors = sel_probs - is_gold.to(sel_probs.dtype)
        # Zero out padding tokens so they don't contribute to sketch
        sel_errors = sel_errors * valid_mask.unsqueeze(1).float()

        s_e = torch.zeros(N, self.m, dtype=torch.float32, device=logits_flat.device)
        sel_signs = self.sign_e[idx]
        sel_buckets = self.bucket_e[idx]

        weighted = (sel_signs.float() * sel_errors).reshape(-1)
        buckets_flat = sel_buckets.reshape(-1)
        batch_idx = (
            torch.arange(N, device=s_e.device)
            .unsqueeze(1)
            .expand(-1, idx.size(1))
            .reshape(-1)
        )
        s_e.index_put_((batch_idx, buckets_flat), weighted, accumulate=True)

        fft_h = torch.fft.fft(s_h, dim=1)
        fft_e = torch.fft.fft(s_e, dim=1)
        z_tokens = torch.fft.ifft(fft_h * fft_e, dim=1).real

        z = z_tokens.sum(dim=0)
        z_norm = z.norm()

        if z_norm < 1e-8:
            return torch.zeros(self.m, dtype=torch.float32, device=z.device)

        return z / z_norm

    def novelty(self, z: torch.Tensor) -> float:
        """Compute directional novelty from normalized sketch."""
        W = self.sketch.get_basis()

        if W.shape[1] == 0:
            return 1.0

        if z.norm().item() < 1e-8:
            return 0.0

        proj = W.T @ z
        redundancy = (proj @ proj).item()
        return 1.0 - torch.clamp(torch.tensor(redundancy), min=0.0, max=1.0).item()

    def accept(self, novelty: float, global_step: int) -> bool:
        """
        Accept if novelty exceeds dynamic threshold.
        During burn-in, accept all updates.
        After burn-in, threshold adapts to maintain target acceptance rate.
        """
        if global_step < self.burn_in:
            return True

        return novelty >= self.current_novelty_threshold

    def update(self, z: torch.Tensor, count_increment: float = 1.0) -> None:
        """Update streaming basis with normalized sketch."""
        self.sketch.update(z)
        self.accepted_count += count_increment

    def step(self, global_step: int) -> None:
        """Adapt threshold to maintain target acceptance rate."""
        if global_step > self.burn_in:
            current_rate = self.acceptance_rate(global_step)
            error = current_rate - self.target_accept_rate
            self.current_novelty_threshold += self.controller_lr * error

    def acceptance_rate(self, global_step: int) -> float:
        """Compute overall acceptance rate."""
        if global_step == 0:
            return 1.0
        return self.accepted_count / global_step

    def reset(self) -> None:
        """Reset gate to initial state."""
        self.sketch.reset()
        self.accepted_count = 0
