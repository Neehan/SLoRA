# SLoRA: Subspace-Gated LoRA

**Token-efficient fine-tuning via directional gradient novelty gating.**

## Overview

SLoRA is a research framework for reducing redundant optimizer updates during LoRA fine-tuning. It gates updates based on **gradient novelty** relative to a learned low-rank subspace of previously accepted gradients, improving sample/token efficiency without sacrificing final performance.

### Research Motivation

Fine-tuning often involves redundant updates—gradients that lie in the span of previously seen directions provide diminishing returns. SLoRA identifies and skips these updates by:

1. Maintaining a streaming sketch of the gradient subspace using Frequent Directions
2. Computing directional novelty of each gradient relative to this subspace
3. Accepting only updates that exceed a novelty threshold

This approach differs fundamentally from loss-based methods (e.g., Selective Backprop) by focusing on gradient geometry rather than loss magnitude.

---

## Algorithm

**Head-gradient proxy gating (gates BEFORE backward):**

1. **Forward pass**: Collect hidden states `h_t ∈ R^H` (pre-classifier) and logits `p_t ∈ R^V` for each token
2. **Error vector**: Compute `e_t = softmax(p_t) - y_t` (one-hot label), sparsified to top-K logits + gold (K=64)
3. **TensorSketch**: For each token, sketch outer product `h_t ⊗ e_t` via:
   - CountSketch `h_t` → `s_h ∈ R^m`
   - CountSketch `e_t` → `s_e ∈ R^m`
   - Convolve via FFT: `z_t = IFFT(FFT(s_h) ⊙ FFT(s_e))`
   - Accumulate: `z = Σ_t z_t`, then normalize `ẑ = z/|z|`
4. **Streaming basis**: Maintain rank-k orthonormal basis `W ∈ R^{m×k}` (k=64) of accepted sketches via Frequent Directions
5. **Novelty score**: `nov = 1 - |W^T ẑ|²` (directional redundancy, in [0,1])
6. **Gate decision**: Accept with probability `σ(β(nov - τ))` where τ adapts via EMA to maintain target acceptance rate
7. **Accept**: Run backward + optimizer step, update basis `W ← FD(W, ẑ)`
8. **Reject**: Skip backward entirely, zero grads
9. **Burn-in**: Always accept first S steps (2k-3k) to initialize basis

**Key advantage:** Skips backward pass on redundant batches using only forward quantities (h, p, y). Cost: O(B·T·(H + K + m log m)) vs O(B·T·H·V) for full head gradient.

**Intuition:** Head gradient `∇_W L = Σ h ⊗ e` captures error-weighted activations. Its subspace novelty correlates with full-network gradient novelty well enough to filter obvious redundancies pre-backward.

---

## Project Structure

```
SLoRA/
├── slora/                      # Python package
│   ├── gate.py                 # HeadGradientGate (TensorSketch + FD)
│   ├── sketch.py               # FrequentDirections streaming sketch
│   ├── trainers/
│   │   └── slora_trainer.py    # HF Trainer with pre-backward gating
│   └── utils/                  # logging, seeding, timing
├── configs/
│   ├── quick_gemma3_1b.yaml    # Quick test (100k samples, 48h)
│   ├── full_gemma3_12b.yaml    # Full run (400k samples, QLoRA)
│   ├── baseline.yaml           # Baseline LoRA (no gating)
│   └── accelerate_config.yaml  # Multi-GPU config
├── scripts/
│   ├── train_slora.py          # Main training script
│   ├── quick_test.sh           # SLURM launcher (baseline + SLoRA)
│   ├── full_gemma_12b.sh       # Full experiment SLURM script
│   └── analyze_results.py      # Compare baseline vs SLoRA
├── data/                       # Dataset cache (auto-created)
├── reports/                    # Analysis outputs
├── requirements.txt
├── environment.yml
└── setup.py
```

---

## Setup

### Option 1: Conda (recommended for SLURM clusters)

```bash
conda env create -f environment.yml
conda activate slora
```

### Option 2: pip

```bash
pip install -e .
```

**Requirements:**
- Python ≥3.10
- PyTorch ≥2.1 with CUDA 12.1
- 4-8× GPUs (L40/A100) depending on model size
- Packages: transformers, accelerate, peft, bitsandbytes, datasets, wandb

---

## Quick Start

### 1. Quick Test (48h, 100k samples, Gemma-2-2B)

**Goal:** Validate that SLoRA reaches baseline performance with ≥30% fewer accepted steps.

```bash
# Via SLURM (runs baseline + SLoRA sequentially)
sbatch scripts/quick_test.sh

# Or run directly
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/baseline.yaml

accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/quick_gemma3_1b.yaml
```

**Pass criteria:** SLoRA validation loss ≤ +0.5% of baseline at ≥30% fewer accepted steps.

### 2. Analyze Results

```bash
python scripts/analyze_results.py \
    --baseline outputs/baseline_gemma_4b \
    --slora outputs/quick_gemma_4b \
    --output reports/quick_analysis.md
```

Check W&B project `slora` for detailed metrics:
- `gate/novelty`: per-step novelty scores
- `gate/acceptance_rate`: running acceptance rate
- `train_loss` vs `gate/accepted_steps`: token efficiency

### 3. Full Experiment (96h, 400k samples, Gemma-2-9B + QLoRA)

```bash
sbatch scripts/full_gemma_12b.sh
```

---

## Usage

### Drop-in API (minimal)

```python
from slora.gate import HeadGradientGate

# Initialize gate
gate = HeadGradientGate(
    d_hidden=4096,           # Model hidden size
    vocab_size=50257,        # Tokenizer vocab size
    m=512,                   # TensorSketch dimension
    k=64,                    # Rank of streaming basis
    target_novelty=0.30,        # Target acceptance rate
    target_novelty_scaler=10.0, # Sigmoid steepness β
    novelty_ema_rate=0.01,    # EMA learning rate for threshold adaptation
    burn_in=2000,            # Always accept first N steps
    seed=0,
    device='cuda',
    reorth_every=128,
    k_topk=64,               # Top-K logits for sparse sketch
)

# Inside training loop (pre-backward gating)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[-1]  # (B, T, H)
    logits = outputs.logits         # (B, T, V)
    labels = inputs['labels']       # (B, T)

    z = gate.embed(h, logits, labels)
    novelty = gate.novelty(z)
    accept = gate.accept(novelty)

if accept:
    outputs = model(**inputs)  # Re-run forward with grad
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    gate.update(z)

optimizer.zero_grad()
gate.step()

print(f"Acceptance rate: {gate.acceptance_rate():.2%}")
```

### HF Trainer Integration

```python
from slora import SLoRATrainer

gate_config = {
    "m": 512,
    "k": 64,
    "target_novelty": 0.30,
    "target_novelty_scaler": 10.0,
    "novelty_ema_rate": 0.01,
    "burn_in": 2000,
    "seed": 0,
    "reorth_every": 128,
    "k_topk": 64,
}

trainer = SLoRATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    gate_config=gate_config,
)

trainer.train()
```

---

## Configuration

All configs are in `configs/*.yaml`. Key parameters:

### Model
- `model.name`: HF model ID (e.g., `google/gemma-3-1b-it`)
- `model.load_in_4bit`: Enable QLoRA (4-bit quantization)
- `model.use_flash_attention_2`: Use Flash Attention 2

### LoRA
- `lora.r`: LoRA rank (default 8)
- `lora.lora_alpha`: LoRA alpha (default 16)
- `lora.target_modules`: List of modules to adapt

### SLoRA Gate
- `slora.enable`: Enable/disable gating (baseline if False)
- `slora.m`: TensorSketch dimension (512, must be power of 2 for FFT)
- `slora.k`: Rank of streaming basis (64)
- `slora.k_topk`: Top-K logits for sparse error sketch (64, 32-128 range)
- `slora.target_novelty`: Target acceptance rate (0.2-0.4, default 0.30)
- `slora.target_novelty_scaler`: Sigmoid steepness β (5-20, default 10.0)
- `slora.novelty_ema_rate`: EMA learning rate for threshold adaptation (0.005-0.02, default 0.01)
- `slora.burn_in`: Steps to always accept (2000-3000)
- `slora.reorth_every`: Re-orthonormalize sketch frequency (128)

### Data
- `data.dataset_name`: HF dataset (e.g., `allenai/tulu-3-sft-mixture`)
- `data.train_split`: Training split with slice (e.g., `train[:100000]`)
- `data.max_seq_length`: Max sequence length (2048-4096)

---

## Hyperparameter Tuning

**Start with defaults:**
- `target_novelty=0.30`, `target_novelty_scaler=10.0`, `novelty_ema_rate=0.01`, `m=512`, `k=64`, `burn_in=2000`

**If acceptance rate doesn't converge to target:**
- Increase `novelty_ema_rate` to 0.02 (faster adaptation)
- Decrease `target_novelty_scaler` to 5.0 (softer sigmoid)

**If acceptance is too noisy:**
- Decrease `novelty_ema_rate` to 0.005 (slower, more stable)
- Increase `target_novelty_scaler` to 15-20 (sharper sigmoid)

**If early instability:**
- Increase `burn_in` to 3000-5000

---

## What to Measure

### Primary (token efficiency)
- **Validation loss** vs **accepted steps** (not total steps)
- Final loss within 0.5% of baseline at fewer accepted steps = success

### Secondary (optional)
- Wall-clock time (add loss-floor gate for skip-backward speedup)
- Downstream task performance (MT-Bench, AlpacaEval)

### Ablations
1. Baseline LoRA (no gating)
2. SLoRA (novelty only)
3. Loss-floor only (selective backprop style)
4. SLoRA + loss-floor (combined)

---

## Logging & Monitoring

Metrics logged to W&B:
- `gate/novelty`: Per-step novelty score
- `gate/accept`: Binary accept/reject (1/0)
- `gate/acceptance_rate`: Running acceptance rate
- `gate/accepted_steps`: Total optimizer steps taken
- `gate/total_steps`: Total batches processed
- `train_loss`, `eval_loss`: Standard training metrics

**Verify correctness:**
- Novelty should be ~1.0 early in training (basis is empty)
- Novelty decreases over time as basis fills
- Acceptance rate stabilizes after burn-in
- Accepted steps << total steps

---

## Troubleshooting

**No LoRA gradients found:**
- Check that PEFT model is loaded correctly
- Verify `lora.target_modules` matches model architecture

**Acceptance rate = 100%:**
- Basis not updating or threshold too low
- Check `gate/novelty` is decreasing over time
- Decrease `target_novelty` (target rate) or increase `target_novelty_scaler`

**OOM errors:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing: true`
- Use QLoRA: `model.load_in_4bit: true`

**Slow training:**
- Check `dataloader_num_workers` is set (default 4)
- Enable `tf32: true` for Ampere GPUs
- Use Flash Attention 2 if supported

---

## Computational Complexity

**Per-step overhead:**
- CountSketch h and e: O(B·T·(H + K)) where K=k_topk (top-K logits)
- FFT convolution: O(B·T·m log m) = O(B·T·512·9) ≈ 4.6k FLOPs per token
- Novelty computation: O(m·k) = O(512·64) ≈ 32k FLOPs per batch
- Sketch update (accepted only): O(m²·k) amortized

**Total gating overhead:** ~5k FLOPs per token (<<0.5% of forward pass)

**Memory:** O(m·k) = 512·64·4 bytes = 128KB (trivial)

**Efficiency gains:**
1. **Backward skip:** Rejected batches avoid full backward pass (main win)
2. **Fewer optimizer steps:** Reduced momentum/Adam state updates
3. **Token efficiency:** Same performance with less data
4. **Sparse error sketch:** O(B·T·K) vs O(B·T·V) for full softmax (K=64 vs V=50k+)

---

## Research Roadmap

**Immediate (validation):**
1. Baseline comparison: verify ≥30% token efficiency improvement at comparable loss
2. Ablation studies: novelty-only vs loss-floor vs combined
3. Hyperparameter sensitivity: target_novelty, target_novelty_scaler, k, burn_in

**Next steps (if validation succeeds):**
1. Scale to 7B-13B models with QLoRA
2. Multiple datasets: instruction tuning, domain adaptation
3. Downstream evaluation: MT-Bench, AlpacaEval, MMLU
4. Loss-floor gate for wall-clock speedup

**Out of scope (keep simple):**
- Multi-metric scoring functions
- Learnable projections or adaptive bases
- Heavy infrastructure until core efficiency is proven

---

## Citation

```bibtex
@misc{slora2024,
  title={SLoRA: Subspace-Gated LoRA for Token-Efficient Fine-Tuning},
  author={Adib Hasan},
  year={2024},
}
```

---

## License

MIT (see LICENSE file)
