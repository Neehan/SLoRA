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

**One metric, one threshold:**

1. **Random projection**: Compress LoRA gradient `g ∈ R^d` → `z = R^T g ∈ R^m` using fixed random signs `R ∈ {±1}^{d×m}` (m=512)
2. **Streaming basis**: Maintain rank-k orthonormal basis `W ∈ R^{m×k}` (k=64) of accepted gradients via Frequent Directions
3. **Novelty score**: `nov = 1 - |W^T ẑ|²` where `ẑ = z/|z|` (directional, in [0,1])
4. **Gate**: Accept update if `nov ≥ tau_n` (default 0.30)
5. **Burn-in**: Always accept first S steps (2k-3k) to initialize basis

**Key insight:** Two batches with identical loss can have different novelty—one may be redundant (in-span of prior gradients) while the other explores new directions.

---

## Project Structure

```
SLoRA/
├── slora/                      # Python package
│   ├── gate.py                 # SubspaceGate class
│   ├── sketch.py               # FrequentDirections streaming sketch
│   ├── hooks.py                # LoRA gradient extraction
│   ├── trainers/
│   │   └── slora_trainer.py    # HF Trainer subclass
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
from slora import SubspaceGate
from slora.hooks import lora_grad_vec

# Initialize gate
gate = SubspaceGate(
    d_lora=1024,      # Flattened LoRA param dimension
    m=512,            # Random projection dimension
    k=64,             # Rank of streaming basis
    tau_n=0.30,       # Novelty threshold
    burn_in=2000,     # Always accept first N steps
    seed=0,
)

# Inside training loop
loss.backward()
g = lora_grad_vec(model)  # Extract flattened LoRA grads

if gate.accept(g):
    optimizer.step()
    gate.update(g)

optimizer.zero_grad()
gate.step()

# Monitor
print(f"Acceptance rate: {gate.acceptance_rate():.2%}")
```

### HF Trainer Integration

```python
from slora import SLoRATrainer

gate_config = {
    "m": 512,
    "k": 64,
    "tau_n": 0.30,
    "burn_in": 2000,
    "seed": 0,
    "reorth_every": 128,
}

trainer = SLoRATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    gate_config=gate_config,
    enable_gate=True,  # Set False for baseline
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
- `slora.m`: Random projection dimension (512)
- `slora.k`: Rank of streaming basis (64)
- `slora.tau_n`: Novelty threshold (0.20-0.50, default 0.30)
- `slora.burn_in`: Steps to always accept (2000-3000)
- `slora.reorth_every`: Re-orthonormalize sketch frequency (128)

### Data
- `data.dataset_name`: HF dataset (e.g., `allenai/tulu-3-sft-mixture`)
- `data.train_split`: Training split with slice (e.g., `train[:100000]`)
- `data.max_seq_length`: Max sequence length (2048-4096)

---

## Hyperparameter Tuning

**Start with defaults:**
- `tau_n=0.30`, `m=512`, `k=64`, `burn_in=2000`

**If acceptance rate too low (<30%):**
- Decrease `tau_n` to 0.20-0.25
- Increase `k` to 96

**If acceptance rate too high (>70%):**
- Increase `tau_n` to 0.35-0.50

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
- `tau_n` too low or basis not updating
- Check `gate/novelty` is decreasing over time
- Increase `tau_n` or check for NaN gradients

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
- Random projection: O(d × m) = O(d × 512)
- Novelty computation: O(m × k) = O(512 × 64) ≈ 32k FLOPs
- Sketch update: O(m² × k) amortized over k steps

**Total:** Negligible compared to backpropagation (<<0.1% training time)

**Memory:** O(m × k) = 512 × 64 × 4 bytes = 128KB (trivial)

**Efficiency gains:**
1. Fewer optimizer steps → reduced momentum/Adam state updates
2. Token efficiency → same performance with less data
3. Optional loss-floor gate can skip backward pass entirely (future work)

---

## Research Roadmap

**Immediate (validation):**
1. Baseline comparison: verify ≥30% token efficiency improvement at comparable loss
2. Ablation studies: novelty-only vs loss-floor vs combined
3. Hyperparameter sensitivity: tau_n, k, burn_in

**Next steps (if validation succeeds):**
1. Scale to 7B-13B models with QLoRA
2. Multiple datasets: instruction tuning, domain adaptation
3. Adaptive threshold controller for target acceptance rate
4. Downstream evaluation: MT-Bench, AlpacaEval, MMLU
5. Loss-floor gate for wall-clock speedup

**Out of scope (keep simple):**
- Multi-metric scoring functions
- Learnable projections or adaptive bases
- Heavy infrastructure until core efficiency is proven

---

## Citation

```bibtex
@misc{slora2024,
  title={SLoRA: Subspace-Gated LoRA for Token-Efficient Fine-Tuning},
  author={},
  year={2024},
}
```

---

## License

MIT (see LICENSE file)
