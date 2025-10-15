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
2. **Error vector**: Compute `e_t = softmax(p_t) - y_t` (one-hot label), sparsified to top-K logits + gold (K=64, deduplicated)
3. **TensorSketch**: For each token, sketch outer product `h_t ⊗ e_t` via:
   - CountSketch `h_t` → `s_h ∈ R^m`
   - CountSketch `e_t` → `s_e ∈ R^m`
   - Convolve via FFT: `z_t = IFFT(FFT(s_h) ⊙ FFT(s_e))`
   - Accumulate: `z = Σ_t z_t`, then normalize `ẑ = z/|z|`
4. **Streaming basis**: Maintain rank-k orthonormal basis `W ∈ R^{m×k}` (k=64) of accepted sketches via Frequent Directions
5. **Novelty score**: `nov = 1 - |W^T ẑ|²` (directional redundancy, in [0,1])
6. **Stochastic gate**: Accept with probability `σ(20·(nov - τ))` where τ adapts via short-memory EMA (decay=0.95) to maintain target acceptance rate
7. **Accept**: Run backward + optimizer step, update basis `W ← FD(W, ẑ)`
8. **Reject**: Skip backward, preserve accumulated gradients
9. **Burn-in**: Always accept first S steps (200-3000) to initialize basis
10. **Threshold clamping**: Keep τ ∈ [0, 1] to prevent runaway

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
│   ├── quick_gemma3_1b_it.yaml # Quick test (1k samples)
│   ├── full_gemma3_12b_it.yaml # Full run (400k samples, QLoRA)
│   └── accelerate_config.yaml  # Multi-GPU config
├── scripts/
│   ├── train_slora.py          # Main training script
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
- 4× GPUs (L40/A100) for DDP training
- Packages: transformers, accelerate, peft, bitsandbytes, datasets, wandb

---

## Quick Start

### 1. Quick Test (1k samples, Gemma-3-1B)

**Goal:** Validate that SLoRA reaches baseline performance with ≥60% fewer accepted steps (40% accept rate).

```bash
# Run with accelerate
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/quick_gemma3_1b_it.yaml
```

**Pass criteria:** SLoRA validation loss ≤ +0.5% of baseline at ≥60% fewer accepted steps.

### 2. Analyze Results

```bash
python scripts/analyze_results.py \
    --baseline outputs/baseline_gemma3_1b_it \
    --slora outputs/quick_gemma3_1b_it \
    --output reports/quick_analysis.md
```

Check W&B project `slora` for detailed metrics:
- `gate/novelty`: per-step novelty scores
- `gate/acceptance_rate`: running acceptance rate (EMA-based, short memory)
- `gate/current_novelty_threshold`: adaptive threshold
- `train_loss` vs `gate/accepted_steps`: token efficiency

### 3. Full Experiment (400k samples, Gemma-3-12B + QLoRA)

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/full_gemma3_12b_it.yaml
```

---

## Usage

### Drop-in API (minimal)

```python
from slora.gate import HeadGradientGate

# Initialize gate
gate = HeadGradientGate(
    d_hidden=4096,              # Model hidden size
    vocab_size=50257,           # Tokenizer vocab size
    m=512,                      # TensorSketch dimension
    k=64,                       # Rank of streaming basis
    target_accept_rate=0.40,    # Target acceptance rate
    initial_threshold=0.05,     # Initial threshold
    controller_lr=0.01,         # Controller learning rate
    burn_in=200,                # Always accept first N steps
    seed=0,
    device='cuda',
    reorth_every=50,
    k_topk=64,                  # Top-K logits for sparse sketch
)

# Inside training loop (pre-backward gating)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[-1]  # (B, T, H)
    logits = outputs.logits         # (B, T, V)
    labels = inputs['labels']       # (B, T)

    z = gate.embed(h, logits, labels)
    novelty = gate.novelty(z)
    accept = gate.accept(novelty, global_step=step)

if accept:
    # Re-run forward with grad (or reuse if single forward)
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    gate.update(z, count_increment=1.0)

gate.step(step, accept)
optimizer.zero_grad()

print(f"Acceptance rate: {gate.acceptance_rate_ema:.2%}")
```

### HF Trainer Integration

```python
from slora import SLoRATrainer

gate_config = {
    "m": 512,
    "k": 64,
    "target_accept_rate": 0.40,
    "initial_threshold": 0.05,
    "controller_lr": 0.01,
    "burn_in": 200,
    "seed": 0,
    "reorth_every": 50,
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
- `slora.k_topk`: Top-K logits for sparse error sketch (64)
- `slora.target_accept_rate`: Target acceptance rate (0.3-0.5, default 0.40)
- `slora.initial_threshold`: Initial novelty threshold (0.05)
- `slora.controller_lr`: Integral controller learning rate (0.01)
- `slora.burn_in`: Steps to always accept (200-3000)
- `slora.reorth_every`: Re-orthonormalize sketch frequency (50-128)

**Key changes from old config:**
- Removed `novelty_sensitivity` (now fixed sigmoid steepness=20)
- Removed `novelty_ema` and `novelty_ema_decay` (not used)
- Added `initial_threshold` (replaces hardcoded 0.5)
- `target_accept_rate` now 0.40 (was 0.70) for real speedup

### Data
- `data.dataset_name`: HF dataset (e.g., `allenai/tulu-3-sft-mixture`)
- `data.train_split`: Training split with slice (e.g., `train[:1000]`)
- `data.max_seq_length`: Max sequence length (256-4096)

---

## Hyperparameter Tuning

**Start with defaults:**
- `target_accept_rate=0.40`, `controller_lr=0.01`, `initial_threshold=0.05`, `m=512`, `k=64`, `burn_in=200`

**If acceptance rate doesn't converge to target:**
- Increase `controller_lr` to 0.02 (faster adaptation)
- Adjust `initial_threshold` closer to expected mean novelty

**If early instability:**
- Increase `burn_in` to 3000 for larger models

**If you want more aggressive gating:**
- Lower `target_accept_rate` to 0.30 (70% rejection)

---

## Implementation Details

### Gradient Accumulation Correctness
- Rejected steps skip backward but **preserve accumulated gradients**
- No `zero_grad()` on reject → multi-step accumulation works correctly
- Optimizer steps normally after accumulation window (even if last micro-batch rejected)

### DDP Loss Handling
- Loss is computed once per step (single forward pass)
- Accepted: contribute full loss to DDP average
- Rejected: contribute 0.0 to DDP average
- Manual all_reduce → average across GPUs for fair logging

### Acceptance Rate Tracking
- Short-memory EMA (decay=0.95, ~20-step window)
- Each GPU increments count by `1/num_processes` to avoid 4× inflation
- Controller uses EMA not cumulative rate → responds quickly to distribution changes

### Gold Token Deduplication
- Check if gold token already in top-K logits
- If duplicate, pad with first top-K token instead of adding gold twice
- Prevents softmax skew in TensorSketch

### Threshold Stability
- Clamp `current_novelty_threshold` to [0, 1] after each update
- Prevents runaway divergence

---

## What to Measure

### Primary (token efficiency)
- **Validation loss** vs **accepted steps** (not total steps)
- Final loss within 0.5% of baseline at fewer accepted steps = success

### Secondary (optional)
- Wall-clock time (expect ~1.5× speedup at 40% accept rate due to backward skip)
- Downstream task performance (MT-Bench, AlpacaEval)

### Ablations
1. Baseline LoRA (no gating)
2. SLoRA (novelty only)
3. Different acceptance rates (0.3, 0.4, 0.5)

---

## Logging & Monitoring

Metrics logged to W&B:
- `gate/novelty`: Per-step novelty score
- `gate/accept`: Binary accept/reject (1/0)
- `gate/acceptance_rate`: Cumulative acceptance rate
- `gate/acceptance_rate_ema`: Short-memory EMA (actual controller input)
- `gate/current_novelty_threshold`: Adaptive threshold
- `gate/accepted_steps`: Total optimizer steps taken
- `gate/total_steps`: Total batches processed
- `train_loss`, `eval_loss`: Standard training metrics (only accepted steps contribute)

**Verify correctness:**
- Novelty should be ~1.0 early in training (basis is empty)
- Novelty decreases over time as basis fills
- Acceptance rate EMA stabilizes around target after burn-in
- Threshold adapts smoothly, stays in [0, 1]
- Train loss matches baseline (not 4× inflated or deflated)

---

## Troubleshooting

**Acceptance rate = 100%:**
- Basis not updating or threshold too low
- Check `gate/novelty` is decreasing over time
- Decrease `target_accept_rate` or increase `burn_in`

**Acceptance rate oscillates wildly:**
- Controller overshooting
- Decrease `controller_lr` to 0.005

**Loss 4× too high or too low:**
- DDP loss aggregation issue
- Verify manual all_reduce in `training_step`

**Gradient accumulation broken:**
- Check rejected steps don't call `zero_grad()`

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
1. Baseline comparison: verify ≥60% token efficiency improvement at comparable loss
2. Ablation studies: different acceptance rates (0.3, 0.4, 0.5)
3. Hyperparameter sensitivity: controller_lr, burn_in, threshold init

**Next steps (if validation succeeds):**
1. Scale to 7B-13B models with QLoRA
2. Multiple datasets: instruction tuning, domain adaptation
3. Downstream evaluation: MT-Bench, AlpacaEval, MMLU

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
