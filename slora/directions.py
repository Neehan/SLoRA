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

    def __init__(
        self, m: int, k: int, reorth_every: int, device: str, dtype: torch.dtype, decay: float = 1.0
    ):
        """
        Initialize empty sketch.

        Args:
            m: Dimension of input vectors
            k: Target rank (number of basis vectors to maintain)
            reorth_every: Re-orthonormalize via QR every N updates (for stability)
            device: Device for tensors
            dtype: Data type for W (must be fp32 or fp64 for stable SVD)
            decay: Exponential decay factor for old directions (1.0 = no decay, <1.0 = forgetting)
        """
        assert dtype in (
            torch.float32,
            torch.float64,
        ), f"dtype must be fp32 or fp64 for stable SVD, got {dtype}"
        assert k <= m, f"k={k} must be <= m={m}"
        assert 0.0 < decay <= 1.0, f"decay must be in (0, 1], got {decay}"
        self.m = m
        self.k = k
        self.reorth_every = reorth_every
        self.device = device
        self.dtype = dtype
        self.decay = decay

        # W starts empty, grows to (m, k) as we add vectors
        self.W = torch.empty(m, 0, dtype=dtype, device=device)
        self.W_T = torch.empty(0, m, dtype=dtype, device=device)
        self.update_count = 0

    def update(self, z: torch.Tensor) -> None:
        """
        Add vector z ∈ R^m to basis using momentum blending.

        Momentum blending: W_t = γ * W_{t-1} + (1-γ) * FD_update(z_t)
        Achieved by decaying singular values (importance) of existing basis.

        Process:
        1. Decay importance of existing directions via singular values: σ ← γ·σ
        2. Append new vector with full importance
        3. SVD to rebalance and keep top k directions

        Args:
            z: Vector ∈ R^m to add to subspace
        """
        z = z.to(dtype=self.dtype, device=self.device)
        assert (
            z.ndim == 1 and z.shape[0] == self.m
        ), f"Expected shape ({self.m},), got {z.shape}"

        # Step 1: Decay existing basis importance via singular values
        if self.W.shape[1] == 0:
            # First vector: no existing basis to decay
            self.W = z.unsqueeze(1)
        else:
            # Get SVD of existing basis
            U, S, Vt = torch.linalg.svd(self.W, full_matrices=False)

            # Decay singular values (importance of old directions)
            S_decayed = self.decay * S

            # Reconstruct basis with decayed importance
            W_decayed = U @ torch.diag(S_decayed) @ Vt

            # Append new vector with full importance
            self.W = torch.cat([W_decayed, z.unsqueeze(1)], dim=1)

        # Step 2: If we exceed rank k, shrink via Frequent Directions
        if self.W.shape[1] > self.k:
            # Compute SVD and keep top k directions
            U, S, Vt = torch.linalg.svd(self.W, full_matrices=False)
            self.W = U[:, : self.k]
        else:
            # Before hitting k, orthonormalize to maintain basis quality
            Q, _ = torch.linalg.qr(self.W)
            self.W = Q

        self.W_T = self.W.T

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
            self.W_T = self.W.T

    def get_basis(self) -> torch.Tensor:
        """
        Return current orthonormal basis W ∈ R^{m×k}.

        Returns:
            W: Orthonormal matrix (columns are orthogonal unit vectors)
        """
        return self.W.clone()

    def get_basis_T(self) -> torch.Tensor:
        """
        Return transpose of current orthonormal basis W^T ∈ R^{k×m}.

        Returns:
            W_T: Transpose of orthonormal matrix
        """
        return self.W_T
