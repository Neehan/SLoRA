# FLoRA: Fractional LoRA via Sketch-Based Token Selection

**Token-efficient fine-tuning via gradient magnitude estimation.**

## Overview

FLoRA reduces training costs by selectively backpropagating only through high-gradient tokens. Instead of updating on all tokens, we:

1. Sketch each token's head gradient using TensorSketch
2. Select top-k% tokens by gradient magnitude
3. Backprop only through selected tokens

This maintains final performance while training on 20-40% of tokens.

---

## Quick Start

### Installation

```bash
conda env create -f environment.yml
conda activate flora
```

**Requirements:**
- Python ≥3.10
- PyTorch ≥2.1
- transformers, accelerate, peft, datasets, wandb

### Run Experiments

**Baseline (standard LoRA):**
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py --config configs/baseline.yaml
```

**Random baseline (random 20% token selection):**
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py --config configs/random_gemma3_1b.yaml
```

**FLoRA (sketch-based selection):**
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py --config configs/quick_gemma3_1b.yaml
```

---

## Configuration

All configs are in `configs/*.yaml`. Three methods available:

### Baseline (method: baseline)
Standard LoRA training on all tokens.

```yaml
gating:
  method: baseline
```

### Random (method: random)
Random token selection baseline.

```yaml
gating:
  method: random
  topk_tokens: 0.2  # Train on 20% of tokens
```

### FLoRA (method: flora)
Sketch-based gradient magnitude selection.

```yaml
gating:
  method: flora
  topk_tokens: 0.2      # Train on top 20% tokens by gradient magnitude
  sketch_dim: 1024      # TensorSketch dimension
  topk_logits: 128      # Number of top logits to sketch (sparse approximation)
```

**Key parameters:**
- `topk_tokens`: Fraction of tokens to backprop (0.2 = 20%)
- `sketch_dim`: Sketch dimension (higher = more accurate, slower)
- `topk_logits`: Sparse error approximation (sketch only top-k logits instead of full vocab)

---

## Project Structure

```
FLoRA/
├── flora/
│   ├── trainers/
│   │   ├── base.py              # Base trainer with token gating
│   │   ├── random_trainer.py   # Random selection baseline
│   │   └── flora_trainer.py    # Sketch-based selection
│   ├── sketch.py                # TensorSketch implementation
│   └── utils/                   # Logging, seeding
├── configs/
│   ├── baseline.yaml            # Standard LoRA
│   ├── random_gemma3_1b.yaml # Random baseline
│   └── quick_gemma3_1b.yaml  # FLoRA
└── scripts/
    ├── train.py                 # Main training script
    ├── quick_baseline.sh        # Run baseline
    ├── quick_random.sh          # Run random
    └── quick_flora.sh           # Run FLoRA
```

---

## Results

Compare methods by checking:
- **Training loss** vs **total tokens seen**
- **Wall-clock time** to target loss
- **Final validation loss**

Expected: FLoRA matches baseline performance while training on 20-40% of tokens.

Check W&B project `flora` for live metrics.

---

## Research Hypothesis

Tokens vary in informativeness. By sketching gradients and selecting high-magnitude tokens, we can maintain performance while drastically reducing compute.

**Baselines to beat:**
1. Random selection (proves it's not just about seeing fewer tokens)
2. Standard LoRA (proves final performance is maintained)

---

## Citation

```bibtex
@misc{flora2024,
  title={FLoRA: Fractional LoRA via Sketch-Based Token Selection},
  author={Adib Hasan},
  year={2024},
}
```

---

## License

MIT
