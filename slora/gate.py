import torch
from slora.sketch import FrequentDirections


class CountSketchProjector:
    """
    Memory-efficient CountSketch projection (O(d) space, not O(d×m)).
    Uses feature hashing: z[h(i)] += s(i) * g[i] for i in [0, d).
    """
    def __init__(self, d: int, m: int, seed: int, device: torch.device):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.bucket = torch.randint(low=0, high=m, size=(d,), generator=g, dtype=torch.long)
        self.sign = (torch.randint(0, 2, (d,), generator=g, dtype=torch.int8) * 2 - 1).to(torch.int8)
        self.m = m
        self.device = device

    @torch.no_grad()
    def project(self, g_vec: torch.Tensor) -> torch.Tensor:
        """Project g_vec: (d,) -> z: (m,) in fp32."""
        z = torch.zeros(self.m, dtype=torch.float32, device=g_vec.device)
        z.index_add_(0, self.bucket.to(g_vec.device), (self.sign.to(g_vec.device).float()) * g_vec.float())
        return z


class SubspaceGate:
    """
    Subspace-Gated Update (SGU) gate for token-efficient fine-tuning.

    **Algorithm Overview:**
    1. Compress LoRA gradient g ∈ R^d to z ∈ R^m via CountSketch projection
       (memory-efficient O(d) space vs dense O(d×m))

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
            reorth_every: Re-orthonormalize W every N accepted updates (128 for stability)
        """
        assert 0.0 <= tau_n <= 1.0, f"tau_n must be in [0,1], got {tau_n}"
        assert k <= m, f"k={k} must be <= m={m}"

        self.d_lora = d_lora
        self.m = m
        self.k = k
        self.tau_n = tau_n
        self.burn_in = burn_in
        self.device = torch.device(device)

        # Use CountSketch projection (O(d) space, not O(d×m))
        self.cs = CountSketchProjector(d_lora, m, seed, self.device)

        # Initialize streaming sketch to track accepted gradient subspace (fp32)
        self.sketch = FrequentDirections(m, k, reorth_every=reorth_every, device=device, dtype=torch.float32)

        # Counters for monitoring
        self.step_count = 0
        self.accepted_count = 0

    @torch.no_grad()
    def embed(self, g_vec: torch.Tensor) -> torch.Tensor:
        """
        Project and normalize gradient to compressed space.

        Args:
            g_vec: Flattened LoRA gradient vector ∈ R^d_lora

        Returns:
            Normalized projection ẑ ∈ R^m (fp32)
        """
        assert g_vec.shape == (self.d_lora,), f"Expected shape ({self.d_lora},), got {g_vec.shape}"
        z = self.cs.project(g_vec)
        return z / (z.norm() + 1e-12)

    @torch.no_grad()
    def novelty(self, z: torch.Tensor) -> float:
        """
        Compute directional novelty from normalized projection.

        Args:
            z: Normalized projection ẑ ∈ R^m

        Returns:
            Novelty score ∈ [0, 1]
        """
        W = self.sketch.get_basis()
        if W.shape[1] == 0:
            return 1.0

        proj = W.T @ z
        redundancy = (proj @ proj).item()
        novelty_score = 1.0 - torch.clamp(torch.tensor(redundancy), min=0.0, max=1.0).item()
        return novelty_score

    @torch.no_grad()
    def accept(self, novelty: float) -> bool:
        """
        Decide whether to accept based on novelty threshold.
        Always accepts during burn-in period.

        Args:
            novelty: Novelty score ∈ [0, 1]

        Returns:
            True if gradient should be accepted
        """
        if self.step_count < self.burn_in:
            return True
        return novelty >= self.tau_n

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """
        Update streaming basis with normalized projection.

        Args:
            z: Normalized projection ẑ ∈ R^m
        """
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
