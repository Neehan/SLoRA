import torch


class TensorSketch:
    """
    Sketch h ⊗ e for tokens via FFT-based convolution.

    Given h ∈ R^d_hidden and e ∈ R^vocab_size, computes sketch(h ⊗ e) ∈ R^sketch_dim
    without materializing the full outer product.

    Uses sparse top-k logits: only sketch e[top_k] instead of full vocab.
    Cost: O(d_hidden + topk_gradients + sketch_dim log sketch_dim) vs O(d_hidden × vocab_size).
    """

    def __init__(self, d_hidden: int, vocab_size: int, sketch_dim: int, topk_gradients: int, seed: int, device: str):
        self.sketch_dim = sketch_dim
        self.topk_gradients = topk_gradients
        self.device = torch.device(device)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.bucket_h = torch.randint(0, sketch_dim, (d_hidden,), generator=gen, dtype=torch.long, device=self.device)
        self.sign_h = torch.randint(0, 2, (d_hidden,), generator=gen, dtype=torch.int8, device=self.device).mul_(2).sub_(1)

        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.bucket_e = torch.randint(0, sketch_dim, (vocab_size,), generator=gen, dtype=torch.long, device=self.device)
        self.sign_e = torch.randint(0, 2, (vocab_size,), generator=gen, dtype=torch.int8, device=self.device).mul_(2).sub_(1)

    def sketch_batch(self, hiddens: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Sketch h_i ⊗ e_i for each token using sparse top-k logits.

        Args:
            hiddens: [N, d_hidden] - pre-head activations
            logits: [N, vocab_size] - classifier outputs
            labels: [N] - target tokens

        Returns:
            [N, sketch_dim] - sketches approximating head gradient h ⊗ e per token
        """
        N = hiddens.shape[0]

        # Get top-k logits to approximate sparse error vector
        topk_values, topk_indices = logits.topk(self.topk_gradients, dim=1)

        # Compute sparse errors directly: e[top_k] = logits[top_k] - one_hot(label)[top_k]
        # Avoid clone() by building errors in-place
        label_mask = (topk_indices == labels.unsqueeze(1))
        errors_sparse = topk_values - label_mask.float()

        # CountSketch hiddens: map h to R^sketch_dim via random hash
        s_h = torch.zeros(N, self.sketch_dim, dtype=torch.float32, device=self.device)
        s_h.index_add_(1, self.bucket_h, self.sign_h.float().unsqueeze(0) * hiddens.float())

        # CountSketch sparse errors: map e[top_k] to R^sketch_dim via random hash
        s_e = torch.zeros(N, self.sketch_dim, dtype=torch.float32, device=self.device)
        sel_signs = self.sign_e[topk_indices]
        sel_buckets = self.bucket_e[topk_indices]

        weighted = (sel_signs.float() * errors_sparse).reshape(-1)
        buckets_flat = sel_buckets.reshape(-1)
        batch_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, self.topk_gradients).reshape(-1)
        s_e.index_put_((batch_idx, buckets_flat), weighted, accumulate=True)

        # Convolve via FFT: sketch(h ⊗ e) = IFFT(FFT(sketch(h)) * FFT(sketch(e)))
        fft_h = torch.fft.fft(s_h, dim=1)
        fft_e = torch.fft.fft(s_e, dim=1)
        z = torch.fft.ifft(fft_h * fft_e, dim=1).real

        # Scale by 1/√sketch_dim to preserve inner products: E[⟨sketch(a⊗b), sketch(a'⊗b')⟩] = ⟨a,a'⟩⟨b,b'⟩
        # Without this, FFT convolution inflates norms by √sketch_dim and inner products by sketch_dim
        z /= self.sketch_dim ** 0.5

        return z
