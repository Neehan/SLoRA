import torch


class TensorSketch:
    """
    TensorSketch for outer products via FFT-based convolution.

    Given vectors a ∈ R^d1 and b ∈ R^d2, computes sketch(a ⊗ b) ∈ R^m without
    materializing the d1×d2 outer product:
    1. CountSketch both: sketch(a), sketch(b) ∈ R^m
    2. Convolve via FFT: sketch(a ⊗ b) = IFFT(FFT(sketch(a)) ⊙ FFT(sketch(b)))

    Cost: O(d1 + d2 + m log m) vs O(d1 × d2) for explicit outer product.
    """

    def __init__(self, d1: int, d2: int, m: int, seed: int, device: torch.device):
        self.m = m
        self.device = device
        self.bucket_1, self.sign_1 = self._init_hash_params(d1, seed)
        self.bucket_2, self.sign_2 = self._init_hash_params(d2, seed + 1)

        # Pre-allocate buffers for sketch computation (reused across batches)
        self._s_h_buffer = None
        self._s_e_buffer = None
        self._current_batch_size = 0

    def _init_hash_params(
        self, dim: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (or broadcast) CountSketch hash buckets/signs so every process
        shares the exact same mappings when running under DDP.
        """
        dist_enabled = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if dist_enabled else 0

        if not dist_enabled or rank == 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            bucket = torch.randint(
                low=0, high=self.m, size=(dim,), generator=generator, dtype=torch.long
            )
            sign = (
                torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int8)
                .mul_(2)
                .sub_(1)
                .to(torch.int8)
            )
        else:
            bucket = torch.empty(dim, dtype=torch.long)
            sign = torch.empty(dim, dtype=torch.int8)

        bucket = bucket.to(self.device)
        sign = sign.to(self.device)

        if dist_enabled:
            torch.distributed.broadcast(bucket, src=0)
            torch.distributed.broadcast(sign, src=0)

        return bucket, sign

    def sketch_batch(
        self,
        dense_vecs: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch sketch h_i ⊗ e_i for all tokens, where e_i is sparse.

        Args:
            dense_vecs: (N, d1) - dense vectors (e.g., hidden states)
            sparse_indices: (N, k) - indices of non-zero entries in sparse vectors
            sparse_values: (N, k) - values at those indices

        Returns:
            z: (N, m) - sketches for each token
        """
        N = dense_vecs.shape[0]

        # Allocate or reuse buffers
        if self._s_h_buffer is None or self._current_batch_size != N:
            self._s_h_buffer = torch.zeros(
                N, self.m, dtype=torch.float32, device=self.device
            )
            self._s_e_buffer = torch.zeros(
                N, self.m, dtype=torch.float32, device=self.device
            )
            self._current_batch_size = N
        else:
            self._s_h_buffer.zero_()
            self._s_e_buffer.zero_()  # type: ignore

        s_h = self._s_h_buffer
        s_e = self._s_e_buffer

        s_h.index_add_(
            1, self.bucket_1, self.sign_1.float().unsqueeze(0) * dense_vecs.float()
        )

        sel_signs = self.sign_2[sparse_indices]
        sel_buckets = self.bucket_2[sparse_indices]

        weighted = (sel_signs.float() * sparse_values).reshape(-1)
        buckets_flat = sel_buckets.reshape(-1)
        batch_idx = (
            torch.arange(N, device=self.device)
            .unsqueeze(1)
            .expand(-1, sparse_indices.size(1))
            .reshape(-1)
        )
        s_e.index_put_((batch_idx, buckets_flat), weighted, accumulate=True)  # type: ignore

        fft_h = torch.fft.fft(s_h, dim=1)
        fft_e = torch.fft.fft(s_e, dim=1)
        z_batch = torch.fft.ifft(fft_h * fft_e, dim=1).real

        # Scale by 1/√m to preserve inner products: E[⟨sketch(a⊗b), sketch(a'⊗b')⟩] = ⟨a,a'⟩⟨b,b'⟩
        # Without this, FFT convolution inflates norms by √m and inner products by m
        z_batch /= self.m**0.5

        return z_batch
