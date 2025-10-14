#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH -A mit_general
#SBATCH --job-name=quick
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=1                      # One task (master launcher)
#SBATCH --cpus-per-task=4              # 4 CPU threads for the task
#SBATCH --gres=gpu:l40s:4                   # Request 4 GPUs
#SBATCH --mem=64GB                     # Total memory
#SBATCH -t 48:00:00                    # 24-hour wall time

# Load your environment
module load miniforge/24.3.0-0

set -e

echo "=== SLoRA Quick Test ==="
echo "Starting time: $(date)"

PROJECT_ROOT="/home/notadib/projects/SLoRA"
cd ${PROJECT_ROOT}

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export WANDB_PROJECT="slora"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4               # keep CPU threads modest
export TRANSFORMERS_ATTENTION_BACKEND=SDPA
export FLASH_ATTENTION_SKIP=True

echo "Running baseline LoRA..."
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/train_slora.py \
    --config configs/baseline.yaml

echo "Baseline complete. Starting SLoRA run..."

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/train_slora.py \
    --config configs/quick_gemma3_1b_it.yaml

echo "Quick test complete!"
echo "End time: $(date)"
echo ""
echo "Compare results:"
echo "  - Baseline: outputs/baseline_gemma3_1b_it"
echo "  - SLoRA:    outputs/quick_gemma3_1b_it"
echo ""
echo "Check W&B project 'slora' for metrics comparison"
