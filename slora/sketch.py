import torch


class FrequentDirections:
    """
    Streaming rank-k sketch via Frequent Directions algorithm.

    **Purpose:**
    Maintain approximate rank-k basis W ∈ R^{m×k} of a stream of vectors z₁, z₂, ...
    such that W captures the top-k principal directions of the vectors seen so far.

    **Algorithm (Liberty 2013):**
    1. Maintain matrix W with ≤ k columns
    2. On new vector z: append z as new column → W has k+1 columns
    3. Compute SVD: W = U Σ Vᵀ
    4. "Shrink" singular values: Σ' = √(Σ² - δ²) where δ = σₖ₊₁
    5. Keep top k columns: W ← U[:, :k] Σ'[:k]

    **Guarantees:**
    - W is approximately orthonormal (enforced via periodic QR)
    - W approximates top-k principal subspace of all vectors seen
    - Space: O(m × k)
    - Time per update: O(m × k²) amortized

    **Why this matters for SLoRA:**
    We stream gradient directions {ẑ₁, ẑ₂, ...} as we accept them.
    W captures the dominant gradient subspace → future gradients projected onto W
    show low novelty if redundant, high novelty if exploring new directions.

    Reference: Liberty (2013) "Simple and Deterministic Matrix Sketching"
    """

    def __init__(self, m: int, k: int, reorth_every: int = 128, device: str = "cuda", dtype: torch.dtype = torch.float32):
        """
        Initialize empty sketch.

        Args:
            m: Dimension of input vectors
            k: Target rank (number of basis vectors to maintain)
            reorth_every: Re-orthonormalize via QR every N updates (for stability)
            device: Device for tensors
            dtype: Data type for W (should match model dtype: float32 or bfloat16)
        """
        assert k <= m, f"k={k} must be <= m={m}"
        self.m = m
        self.k = k
        self.reorth_every = reorth_every
        self.device = device
        self.dtype = dtype

        # W starts empty, grows to (m, k) as we add vectors
        self.W = torch.empty(m, 0, dtype=dtype, device=device)
        self.update_count = 0

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """
        Add unit vector z ∈ R^m to sketch and shrink to rank k via Frequent Directions.

        Process:
        1. Append z as new column: W ← [W | z]  (now has k+1 or fewer columns)
        2. If W has > k columns:
           a. Compute SVD: W = U Σ Vᵀ
           b. Shrink singular values: σᵢ' = √(σᵢ² - δ²) where δ = σₖ₊₁
           c. Keep top k: W ← U[:, :k] diag(Σ'[:k])
        3. Periodically re-orthonormalize via QR for numerical stability

        **Why shrinking works:**
        Subtracting δ² from all squared singular values reduces contribution of
        smallest direction (σₖ₊₁) while preserving relative importance of top-k.
        This is the "frequent directions" trick: discard least frequent direction.

        Args:
            z: Unit vector ∈ R^m (must have norm ≈ 1.0)
        """
        z = z.to(dtype=self.dtype, device=self.device)
        assert z.ndim == 1 and z.shape[0] == self.m, f"Expected shape ({self.m},), got {z.shape}"

        # Step 1: Append new vector as column
        if self.W.shape[1] == 0:
            # First vector: W is just z as a column
            self.W = z.unsqueeze(1)
        else:
            # Concatenate z to existing columns
            self.W = torch.cat([self.W, z.unsqueeze(1)], dim=1)

        # Step 2: If we exceed rank k, shrink via Frequent Directions
        if self.W.shape[1] > self.k:
            # Compute SVD of current matrix
            U, S, Vt = torch.linalg.svd(self.W, full_matrices=False)

            # Shrink: subtract (k+1)-th squared singular value from all
            delta = S[self.k] ** 2  # σₖ₊₁²
            S_new = torch.sqrt(torch.clamp(S[:self.k] ** 2 - delta, min=0.0))

            # Reconstruct with top k vectors and shrunk singular values
            self.W = U[:, :self.k] * S_new.unsqueeze(0)  # (m, k) * (1, k) = (m, k)

        # Step 3: Periodic re-orthonormalization for numerical stability
        self.update_count += 1
        if self.update_count % self.reorth_every == 0:
            self._reorthonormalize()

    @torch.no_grad()
    def _reorthonormalize(self) -> None:
        """
        Re-orthonormalize W via QR decomposition for numerical stability.

        Over many updates, floating-point errors can cause W to lose orthogonality.
        QR decomposition recovers an orthonormal basis: W = QR → W ← Q.
        Called every reorth_every updates (default 128).
        """
        if self.W.shape[1] > 0:
            Q, _ = torch.linalg.qr(self.W)
            self.W = Q

    def get_basis(self) -> torch.Tensor:
        """
        Return current orthonormal basis W ∈ R^{m×k}.

        Returns:
            W: Orthonormal matrix (columns are orthogonal unit vectors)
        """
        return self.W.clone()

    def reset(self) -> None:
        """Reset sketch to empty (clears all accepted vectors)."""
        self.W = torch.empty(self.m, 0, dtype=self.dtype, device=self.device)
        self.update_count = 0
