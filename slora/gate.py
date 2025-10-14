import torch
from slora.sketch import FrequentDirections


class SubspaceGate:
    """
    Subspace-Gated Update (SGU) gate for token-efficient fine-tuning.

    **Algorithm Overview:**
    1. Compress LoRA gradient g ∈ R^d to z ∈ R^m via random projection: z = R^T g
       where R ∈ {±1}^{d×m} is fixed random signs (Johnson-Lindenstrauss projection)

    2. Maintain rank-k orthonormal basis W ∈ R^{m×k} of accepted gradient directions
       using Frequent Directions streaming algorithm

    3. Compute novelty as directional redundancy: nov = 1 - |W^T ẑ|²
       where ẑ = z / |z| is normalized projection
       - nov = 1.0: gradient is orthogonal to all accepted gradients (novel)
       - nov = 0.0: gradient lies in span of accepted gradients (redundant)

    4. Accept gradient if nov ≥ tau_n (threshold), skip optimizer step otherwise

    5. If accepted, update basis W with new direction ẑ

    **Key Properties:**
    - Gates on directional novelty, NOT loss magnitude (unlike Selective Backprop)
    - Two batches with same loss treated differently if one is redundant
    - Burn-in period: always accept first S steps to initialize basis
    - Overhead: O(m·k) ≈ 32k FLOPs per step (negligible vs backprop)
    """

    def __init__(
        self,
        d_lora: int,
        m: int = 512,
        k: int = 64,
        tau_n: float = 0.30,
        burn_in: int = 2000,
        seed: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        reorth_every: int = 128,
    ):
        """
        Initialize subspace gate.

        Args:
            d_lora: Dimension of flattened LoRA gradient vector (auto-computed from model)
            m: Random projection target dimension (512 is good default)
            k: Rank of streaming basis W (64 is good default, k << m)
            tau_n: Novelty threshold ∈ [0, 1] for acceptance (0.3 = accept if 30%+ novel)
            burn_in: Always accept first N steps to initialize basis (2k-3k typical)
            seed: Random seed for projection matrix R (for reproducibility)
            device: Device for tensors ('cuda' or 'cpu')
            dtype: Data type for tensors (should match model: bfloat16 or float32)
            reorth_every: Re-orthonormalize W every N accepted updates (128 for stability)
        """
        assert 0.0 <= tau_n <= 1.0, f"tau_n must be in [0,1], got {tau_n}"
        assert k <= m, f"k={k} must be <= m={m}"

        self.d_lora = d_lora
        self.m = m
        self.k = k
        self.tau_n = tau_n
        self.burn_in = burn_in
        self.device = device
        self.dtype = dtype

        # Create fixed random projection matrix R ∈ {±1}^{d×m}
        # Uses global seed for reproducibility (set via set_seed() in training script)
        torch.manual_seed(seed)
        self.R = torch.sign(torch.randn(d_lora, m, device=device, dtype=dtype))

        # Initialize streaming sketch to track accepted gradient subspace
        self.sketch = FrequentDirections(m, k, reorth_every=reorth_every, device=device, dtype=dtype)

        # Counters for monitoring
        self.step_count = 0
        self.accepted_count = 0

    @torch.no_grad()
    def novelty(self, g_vec: torch.Tensor) -> float:
        """
        Compute directional novelty of gradient relative to accepted subspace.

        Steps:
        1. Project gradient to compressed space: z = R^T g  (d → m dimensions)
        2. Normalize to unit direction: ẑ = z / |z|
        3. Project onto accepted subspace: proj = W^T ẑ  (W is orthonormal basis)
        4. Compute novelty: nov = 1 - |proj|² = 1 - Σᵢ (wᵢ^T ẑ)²

        Interpretation:
        - |proj|² = squared norm of projection = how much of ẑ lies in span(W)
        - nov = 1 - |proj|² = how much of ẑ is orthogonal to span(W)
        - nov = 1.0: gradient is fully novel (orthogonal to all accepted gradients)
        - nov = 0.5: gradient is 50% novel, 50% redundant
        - nov = 0.0: gradient is fully redundant (lies in span of accepted gradients)

        Args:
            g_vec: Flattened LoRA gradient vector ∈ R^d_lora

        Returns:
            Novelty score ∈ [0, 1]
        """
        g_vec = g_vec.to(device=self.device)
        assert g_vec.shape == (
            self.d_lora,
        ), f"Expected shape ({self.d_lora},), got {g_vec.shape}"

        # Step 1: Random projection to compress gradient
        z = self.R.T @ g_vec  # (d,) → (m,)

        # Step 2: Normalize to unit direction (for directional comparison)
        z = z / (z.norm() + 1e-12)  # ẑ

        # Step 3: Get current accepted subspace basis
        W = self.sketch.get_basis()  # (m, k) orthonormal matrix
        if W.shape[1] == 0:
            # Basis is empty (no accepted gradients yet) → fully novel
            return 1.0

        # Step 4: Project onto subspace and compute redundancy
        proj = W.T @ z  # (k,) - coordinates in basis W
        redundancy = (proj @ proj).item()  # |proj|² = Σᵢ (wᵢ^T ẑ)²
        novelty_score = 1.0 - torch.clamp(torch.tensor(redundancy), min=0.0, max=1.0).item()

        return novelty_score

    @torch.no_grad()
    def accept(self, g_vec: torch.Tensor) -> bool:
        """
        Decide whether to accept gradient based on novelty threshold.
        Always accepts during burn-in period.

        Args:
            g_vec: Flattened gradient vector

        Returns:
            True if gradient should be accepted
        """
        if self.step_count < self.burn_in:
            return True

        nov = self.novelty(g_vec)
        return nov >= self.tau_n

    @torch.no_grad()
    def update(self, g_vec: torch.Tensor) -> None:
        """
        Update streaming basis with accepted gradient direction.

        Process:
        1. Project gradient to compressed space: z = R^T g
        2. Normalize: ẑ = z / |z|
        3. Add ẑ to streaming sketch (Frequent Directions updates basis W)

        Should only be called after accept() returns True.

        Args:
            g_vec: Flattened LoRA gradient vector ∈ R^d_lora
        """
        g_vec = g_vec.to(device=self.device)

        # Project and normalize (same as in novelty())
        z = self.R.T @ g_vec
        z = z / (z.norm() + 1e-12)

        # Update streaming sketch with new direction
        self.sketch.update(z)
        self.accepted_count += 1

    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1

    def acceptance_rate(self) -> float:
        """Compute overall acceptance rate."""
        if self.step_count == 0:
            return 1.0
        return self.accepted_count / self.step_count

    def reset(self) -> None:
        """Reset gate to initial state."""
        self.sketch.reset()
        self.step_count = 0
        self.accepted_count = 0
