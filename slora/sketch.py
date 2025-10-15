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
        g1 = torch.Generator(device="cpu").manual_seed(seed)
        self.bucket_1 = torch.randint(
            low=0, high=m, size=(d1,), generator=g1, dtype=torch.long
        ).to(device)
        self.sign_1 = (
            (torch.randint(0, 2, (d1,), generator=g1, dtype=torch.int8) * 2 - 1).to(
                torch.int8
            )
        ).to(device)

        g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.bucket_2 = torch.randint(
            low=0, high=m, size=(d2,), generator=g2, dtype=torch.long
        ).to(device)
        self.sign_2 = (
            (torch.randint(0, 2, (d2,), generator=g2, dtype=torch.int8) * 2 - 1).to(
                torch.int8
            )
        ).to(device)

        self.m = m
        self.device = device

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

        s_h = torch.zeros(N, self.m, dtype=torch.float32, device=self.device)
        s_h.index_add_(
            1, self.bucket_1, self.sign_1.float().unsqueeze(0) * dense_vecs.float()
        )

        s_e = torch.zeros(N, self.m, dtype=torch.float32, device=self.device)
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
        s_e.index_put_((batch_idx, buckets_flat), weighted, accumulate=True)

        fft_h = torch.fft.fft(s_h, dim=1)
        fft_e = torch.fft.fft(s_e, dim=1)
        z_batch = torch.fft.ifft(fft_h * fft_e, dim=1).real

        return z_batch
