import torch


class FrequentDirections:
    """
    Streaming rank-k orthonormal basis via incremental QR/SVD.

    **Purpose:**
    Maintain orthonormal basis W ∈ R^{m×k} spanning the top-k principal directions
    of a stream of unit vectors z₁, z₂, ...

    **Algorithm:**
    1. Maintain matrix W with ≤ k orthonormal columns
    2. On new vector z: append z as new column
    3. If W has ≤ k columns: orthonormalize via QR
    4. If W has k+1 columns: SVD, keep top k left singular vectors (orthonormal)
    5. Periodic re-orthonormalization via QR for numerical stability

    **Properties:**
    - W is always orthonormal: W^T W = I_k
    - |W^T z|² ≤ 1 for any unit vector z (subspace projection bound)
    - Space: O(m × k)
    - Time per update: O(m × k²) for QR

    **Why this matters for SLoRA:**
    We stream accepted gradient sketches {ẑ₁, ẑ₂, ...} (unit vectors).
    W captures the gradient subspace → novelty = 1 - |W^T ẑ|² measures
    how orthogonal a new gradient is to previously accepted directions.

    Note: Simplified from Liberty (2013) Frequent Directions—we drop singular value
    shrinking since we only need subspace span for novelty, not low-rank approximation.
    """

    def __init__(self, m: int, k: int, reorth_every: int = 128, device: str = "cuda", dtype: torch.dtype = torch.float32):
        """
        Initialize empty sketch.

        Args:
            m: Dimension of input vectors
            k: Target rank (number of basis vectors to maintain)
            reorth_every: Re-orthonormalize via QR every N updates (for stability)
            device: Device for tensors
            dtype: Data type for W (must be fp32 or fp64 for stable SVD)
        """
        assert dtype in (torch.float32, torch.float64), f"dtype must be fp32 or fp64 for stable SVD, got {dtype}"
        assert k <= m, f"k={k} must be <= m={m}"
        self.m = m
        self.k = k
        self.reorth_every = reorth_every
        self.device = device
        self.dtype = dtype

        # W starts empty, grows to (m, k) as we add vectors
        self.W = torch.empty(m, 0, dtype=dtype, device=device)
        self.update_count = 0

    def update(self, z: torch.Tensor) -> None:
        """
        Add unit vector z ∈ R^m to orthonormal basis.

        Process:
        1. Append z as new column: W ← [W | z]
        2. If W has ≤ k columns: orthonormalize via QR → W is orthonormal
        3. If W has k+1 columns: SVD, keep top k left singular vectors (U[:, :k])
        4. Periodic re-orthonormalization via QR for numerical stability

        This maintains an orthonormal basis spanning the dominant subspace
        of all vectors seen so far.

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

            # Keep orthonormal basis (drop scaling for novelty computation)
            self.W = U[:, :self.k]
        else:
            # Before hitting k, orthonormalize to maintain basis quality
            Q, _ = torch.linalg.qr(self.W)
            self.W = Q

        # Step 3: Periodic re-orthonormalization for numerical stability
        self.update_count += 1
        if self.update_count % self.reorth_every == 0:
            self._reorthonormalize()

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
