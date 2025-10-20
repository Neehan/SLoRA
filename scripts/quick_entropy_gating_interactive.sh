#!/bin/bash
# === FLoRA Quick Test (interactive 2-GPU) ===
set -e

echo "=== FLoRA Quick Test (2-GPU interactive) ==="
echo "Start time: $(date)"

# Activate your conda/Miniforge env
module load miniforge/24.3.0-0

# Move to project root
cd "$(dirname "$0")/.."

# --- Environment variables ---
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export WANDB_PROJECT="flora"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export TRANSFORMERS_ATTENTION_BACKEND=SDPA
export FLASH_ATTENTION_SKIP=True

# --- Entropy Gating run ---
echo "Running Entropy Gating..."

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 2 \
    --mixed_precision bf16 \
    scripts/train.py \
    --config configs/entropy_gating_gemma3_1b_it.yaml

echo "Entropy Gating test complete!"
echo "End time: $(date)"
echo ""
echo "Results: outputs/entropy_gating_gemma3_1b_it"
echo "Check W&B project 'flora' for metrics"
