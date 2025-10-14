#!/bin/bash
# === SLoRA Quick Test (interactive 2-GPU) ===
set -e

echo "=== SLoRA Quick Test (2-GPU interactive) ==="
echo "Start time: $(date)"

# Activate your conda/Miniforge env
module load miniforge/24.3.0-0

# Move to project root
cd "$(dirname "$0")/.."

# --- Environment variables ---
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export WANDB_PROJECT="slora"
export CUDA_VISIBLE_DEVICES=0,1        # use 2 GPUs only
export OMP_NUM_THREADS=4               # keep CPU threads modest
export TRANSFORMERS_ATTENTION_BACKEND=SDPA
export FLASH_ATTENTION_SKIP=True

# --- Baseline run ---
echo "Running baseline LoRA..."
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 2 \
    --mixed_precision bf16 \
    scripts/train_slora.py \
    --config configs/baseline.yaml

# --- SLoRA run ---
echo "Baseline complete. Starting SLoRA run..."

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 2 \
    --mixed_precision bf16 \
    scripts/train_slora.py \
    --config configs/quick_gemma3_1b_it.yaml

echo "Quick test complete!"
echo "End time: $(date)"
echo
echo "Compare results:"
echo "  - Baseline: outputs/baseline_gemma3_1b_it"
echo "  - SLoRA:    outputs/quick_gemma3_1b_it"
echo
echo "Check W&B project 'slora' for metrics comparison"
