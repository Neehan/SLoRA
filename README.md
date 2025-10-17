# SLoRA: Subspace-Gated LoRA

**Token-efficient fine-tuning via gradient novelty filtering.**

## Overview

SLoRA reduces training compute by filtering redundant samples **before** fine-tuning. It runs a fast inference-only filter pass to identify high-novelty samples, then trains standard LoRA on only the accepted subset.

### Key Idea

Not all training samples contribute equally to learning. Many gradients lie in the span of previously seen directions and provide diminishing returns. SLoRA:

1. **Filter pass**: Run inference on full dataset, compute gradient novelty via head-gradient proxy
2. **Subspace tracking**: Maintain low-rank sketch of accepted gradient directions using Frequent Directions
3. **Novelty gating**: Accept samples whose gradients are directionally novel relative to tracked subspace
4. **Train pass**: Fine-tune LoRA on accepted samples only (typically 25-40% of data)

**Result:** Match baseline performance with 60-75% fewer training samples and ~40% wallclock speedup.

---

## Architecture

**Two-pass training:**

### Pass 1: Filter (inference-only, fast)
- Forward pass through base model (no gradients)
- Extract hidden states `h_t` and logits for each token
- Compute head-gradient proxy: `∇_W L ≈ Σ h_t ⊗ e_t` via TensorSketch
- Track accepted gradient subspace via Frequent Directions
- Gate based on directional novelty: `nov = 1 - ||W^T ẑ||²`
- Output: indices of accepted samples

### Pass 2: Train (standard LoRA)
- Subset dataset to accepted indices only
- Run vanilla LoRA training (HuggingFace Trainer)
- Adjust max_steps to match filtered dataset size (1 epoch over filtered data)

**Why this works:** The head-gradient proxy `∇_W L_head = Σ h_t ⊗ e_t` (outer product of pre-classifier hidden states and error vectors) correlates strongly with full-network gradient novelty but costs only O(d·v) via sketching instead of O(d·v·L) for full backprop.

---

## Algorithm Details

### Head-Gradient Proxy Sketch

1. **Extract forward quantities:**
   - Hidden states: `h_t ∈ R^d` (last layer, pre-classifier)
   - Logits: `p_t ∈ R^v`
   - Labels: `y_t`

2. **Compute sparse error vectors:**
   - `e_t = softmax(p_t) - one_hot(y_t)`
   - Keep only top-k logits + gold label to reduce noise (k=64-128)
   - Batch across all tokens in batch

3. **TensorSketch outer products:**
   - Sketch `h_t ⊗ e_t` without materializing d×v matrix:
     - CountSketch `h_t` → `s_h ∈ R^m`
     - CountSketch `e_t` → `s_e ∈ R^m` (sparse)
     - Convolve via FFT: `z_t = IFFT(FFT(s_h) ⊙ FFT(s_e))`
   - Aggregate: `z = Σ_t z_t / ||Σ_t z_t||` (normalized batch sketch)
   - Cost: O(batch_size × seq_len × (d + k + m log m))

4. **Streaming subspace (Frequent Directions):**
   - Maintain rank-k orthonormal basis `W ∈ R^{m×k}` of accepted sketches
   - Update: append `z`, SVD, shrink smallest singular values
   - Exponential decay on old directions: `W ← √decay · W` before update

5. **Directional novelty:**
   - `nov = 1 - ||W^T ẑ||²` (fraction of `ẑ` orthogonal to subspace)
   - Pure angular measure, independent of gradient magnitude

6. **Adaptive threshold:**
   - Start with burn-in: accept all samples for first N steps to build initial subspace
   - After burn-in: accept if `nov > τ`
   - Adapt `τ` via integral controller to maintain target acceptance rate:
     - `τ ← τ + lr × (acceptance_rate_ema - target_rate)`
   - Clamp `τ ∈ [0, 1]`

### Random Baseline

For ablation, set `random: true` to ignore novelty and accept samples uniformly at random with probability `target_accept_rate`. This isolates the benefit of novelty-based selection vs pure data reduction.

---

## Project Structure

```
SLoRA/
├── slora/
│   ├── gate.py              # HeadGradientGate (novelty computation)
│   ├── sketch.py            # TensorSketch (FFT-based outer product sketch)
│   ├── directions.py        # FrequentDirections (streaming low-rank approximation)
│   ├── filter.py            # filter_pass() (inference loop for dataset filtering)
│   ├── trainer.py           # SLoRATrainer (HF Trainer wrapper with filtering)
│   └── utils/               # logging, seeding, data prep
├── configs/
│   ├── baseline.yaml        # Baseline LoRA (no filtering)
│   └── quick_gemma3_1b_it.yaml  # SLoRA config (inherits from baseline)
├── scripts/
│   ├── train_slora.py       # Main training script
│   └── quick_test.sh        # SLURM job script
└── README.md
```

---

## Setup

### Requirements
- Python ≥3.10
- PyTorch ≥2.0 with CUDA 12.1+
- 4× GPUs (A100/L40) for multi-GPU training
- Packages: transformers, accelerate, peft, datasets, wandb

### Installation

```bash
# Option 1: Conda (recommended for SLURM)
conda env create -f environment.yml
conda activate slora

# Option 2: pip
pip install -e .
```

---

## Quick Start

### 1. Baseline (no filtering)

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/baseline.yaml
```

This trains LoRA on 128k samples (1000 optimizer steps × 128 samples/step) from Tulu-3 SFT mixture.

### 2. SLoRA (with filtering)

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/quick_gemma3_1b_it.yaml
```

**What happens:**
1. **Filter pass** (~15 min): Runs inference on 128k samples, accepts ~25-40% based on novelty
2. **Train pass** (~10 min): Trains LoRA on accepted subset only
3. **Total time**: ~25 min vs ~40 min for baseline (40% speedup)

### 3. Monitor Results

Check W&B project for:
- `filter/gate/acceptance_rate`: Should stabilize at target (e.g., 0.25 or 0.40)
- `filter/gate/novelty_avg`: Novelty over time (spikes at dataset boundaries)
- `eval/loss`: Should match baseline within ~0.5%
- `filter/elapsed_time_seconds`: Filter pass progress

---

## Configuration

Configs use YAML inheritance. `quick_gemma3_1b_it.yaml` inherits all settings from `baseline.yaml` and only overrides:

```yaml
base: baseline.yaml  # Inherit model, training, lora, data settings

training:
  output_dir: ./outputs/quick_gemma3_1b_it

slora:
  enable: true               # Enable filtering
  m: 1024                    # Sketch dimension (power of 2, larger = better quality)
  k: 32                      # Subspace rank (higher = capture more directions)
  target_accept_rate: 0.25   # Target 25% acceptance (75% filtering)
  initial_threshold: 0.05    # Starting novelty threshold
  controller_lr: 0.01        # Threshold adaptation rate
  burn_in: 100               # Accept all for first N steps (build initial subspace)
  seed: 0
  reorth_every: 32           # Re-orthonormalize subspace every N updates
  k_topk: 128                # Top-k logits for sparse sketch
  random: false              # If true: random filtering (ablation baseline)
  subspace_decay: 0.99       # Exponential decay for old subspace directions

logging:
  wandb_run_name: quick_gemma3_1b_it
```

### Key Hyperparameters

**Filtering aggressiveness:**
- `target_accept_rate: 0.40` → accept 40% of data (conservative, safe)
- `target_accept_rate: 0.25` → accept 25% of data (aggressive, more speedup)
- `target_accept_rate: 0.15` → accept 15% of data (very aggressive, may hurt performance)

**Sketch quality vs speed:**
- `m: 512` → 2x faster FFT, lower quality
- `m: 1024` → balanced (default)
- `m: 2048` → higher quality, 2x slower FFT

**Subspace capacity:**
- `k: 32` → fewer directions tracked (default)
- `k: 64` → more capacity, captures more diversity
- `k: 128` → very high capacity (may overfit to noise)

**Adaptation speed:**
- `controller_lr: 0.01` → default
- `controller_lr: 0.02` → faster threshold adaptation (for aggressive targets)
- `burn_in: 100` → 10% of training for initialization
- `burn_in: 50` → faster startup (if subspace stabilizes quickly)

**Memory vs novelty decay:**
- `subspace_decay: 0.99` → slow forgetting (default, good for within-dataset redundancy)
- `subspace_decay: 0.95` → faster forgetting (if data distribution shifts frequently)

---

## Usage: Integrate Into Your Code

### Standalone API

```python
from slora.gate import HeadGradientGate

gate = HeadGradientGate(
    d_hidden=2048,
    vocab_size=50257,
    m=1024,
    k=32,
    target_accept_rate=0.30,
    controller_lr=0.01,
    burn_in=100,
    seed=0,
    device='cuda',
    reorth_every=32,
    k_topk=128,
    initial_threshold=0.05,
    random=False,
    subspace_decay=0.99,
)

# During inference loop (no gradients needed)
with torch.no_grad():
    outputs = model(**batch, output_hidden_states=True)
    h = outputs.hidden_states[-1]  # (B, T, d_hidden)
    logits = outputs.logits         # (B, T, vocab_size)
    labels = batch['labels']        # (B, T)

    z = gate.embed(h, logits, labels)  # Sketch head gradient
    novelty = gate.novelty(z, step)    # Compute novelty score
    accept = gate.accept(novelty, step) # Gate decision

if accept:
    accepted_indices.append(batch_idx)
    gate.update(z, count_increment=1.0)

gate.step(step, accept)  # Adapt threshold
```

### HuggingFace Trainer Integration

The `SLoRATrainer` automatically handles filtering:

```python
from slora.trainer import SLoRATrainer

gate_config = {
    "m": 1024,
    "k": 32,
    "target_accept_rate": 0.30,
    "initial_threshold": 0.05,
    "controller_lr": 0.01,
    "burn_in": 100,
    "seed": 0,
    "reorth_every": 32,
    "k_topk": 128,
    "random": False,
    "subspace_decay": 0.99,
    # Copy training params needed for filter pass
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "max_steps": 1000,
    "logging_steps": 10,
}

trainer = SLoRATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    gate_config=gate_config if enable_slora else None,
)

trainer.train()  # Automatically runs filter pass if gate_config provided
```

---

## Implementation Details

### Filter Pass Optimizations

1. **Buffer pre-allocation**: Sketch buffers reused across batches (10-20% speedup)
2. **Inference mode**: `torch.inference_mode()` disables autograd
3. **Sparse error vectors**: Only compute top-k logits, not full softmax
4. **DDP-safe**: Gradients synced via all_reduce before novelty computation

### Training Pass Adjustments

1. **Dataset subsetting**: `Subset(dataset, accepted_indices)` creates filtered view
2. **Max steps adjustment**: Automatically reduces `max_steps` to match filtered dataset size for fair comparison (1 epoch over filtered data)
3. **Final eval**: Always runs `trainer.evaluate()` after training completes

### Timing Instrumentation

Filter pass logs breakdown:
- `Forward: X.Xs (Y%)` - model inference time
- `Gating: X.Xs (Y%)` - sketch + novelty computation time

Typical: 85-95% forward, 5-15% gating.

---

## Results Interpretation

### Success Criteria

SLoRA is working if:
1. **Acceptance rate stabilizes** at target (±5%) after burn-in
2. **Novelty decays** from ~1.0 to ~0.1-0.2 within first 200-500 steps
3. **Final eval loss** within 0.5% of baseline
4. **Wallclock speedup** ~1.5-2x depending on acceptance rate

### Expected Metrics

**Filter pass (1000 steps, 128k samples):**
- Time: ~15 min on 4×A100
- Acceptance: 25-40% of batches (depending on target)
- Forward: ~90% of time, Gating: ~10%

**Train pass (on accepted subset):**
- Time: ~10-15 min (fewer samples than baseline)
- Eval loss: within 0.5% of baseline

**Total speedup:** ~40-60% wallclock reduction.

### Common Issues

**Acceptance rate = 100%:**
- Novelty not decreasing → subspace not updating
- Increase `burn_in` or check that `gate.update()` is called on accepts

**Acceptance rate too low/high:**
- Controller not converging → increase `controller_lr`
- Wrong target → adjust `target_accept_rate`

**Novelty stays high (~0.5+):**
- Sketch dimensions too small → increase `m` or `k`
- Top-k filtering too aggressive → increase `k_topk`

**Performance degradation:**
- Filtering too aggressive → increase `target_accept_rate` to 0.40
- Random baseline outperforms novelty → check subspace update logic

---

## Performance Tuning

### If filtering is slow:

1. **Reduce sketch dimensions:**
   - `m: 512` (2x faster FFT)
   - `k_topk: 64` (2x faster sparse computation)
   - Combined: ~4x faster gating

2. **Reduce sequence length (filter only):**
   - Filter on `max_seq_length: 256`, train on 512
   - 2x faster forward pass during filtering
   - Trade-off: may miss novelty in long sequences

### If model forward is bottleneck:

1. **Quantization:** Load model with `load_in_8bit: true` (filter pass only)
2. **Smaller batch size:** Reduce `per_device_train_batch_size` during filtering
3. **Gradient checkpointing off:** Already disabled (correct for inference)

---

## Research Notes

### Current Status

- **Implementation:** Two-pass filtering + standard LoRA training
- **Tested on:** Gemma-3-1B-it, Tulu-3 SFT mixture (128k samples)
- **Result:** Matches baseline with 25-40% of data (~40% speedup)

### Design Decisions

**Why two-pass instead of online gating?**
- Simpler implementation (standard HF Trainer for training)
- Easier to debug (separate filter metrics from training metrics)
- Allows offline analysis of filtering decisions
- No complex gradient accumulation logic during rejection

**Why head-gradient proxy?**
- Avoids full backward pass during filtering
- Strong correlation with full-network gradient novelty
- Cheap to compute via TensorSketch

**Why Frequent Directions?**
- Streaming algorithm (no need to store all gradients)
- Provable approximation guarantees
- Simple to implement

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

MIT
