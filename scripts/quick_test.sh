#!/bin/bash
#SBATCH -p mit_preemptable              # Use GPU partition with longer runtime
#SBATCH -A mit_general
#SBATCH --job-name=quick
#SBATCH -N 1                            # Single node
#SBATCH --ntasks=1                      # One task (master launcher)
#SBATCH --cpus-per-task=4              # 4 CPU threads for the task
#SBATCH --gres=gpu:h100:4                   # Request 4 GPUs
#SBATCH --mem=40GB                     # Total memory
#SBATCH -t 24:00:00                    # 24-hour wall time

# Load your environment
module load miniforge/24.3.0-0

set -e

echo "=== SLoRA Quick Test ==="
echo "Starting time: $(date)"

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="slora"
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Running baseline LoRA..."
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/baseline.yaml

echo "Baseline complete. Starting SLoRA run..."

accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/quick_gemma3_4b.yaml

echo "Quick test complete!"
echo "End time: $(date)"
echo ""
echo "Compare results:"
echo "  - Baseline: outputs/baseline_gemma_4b"
echo "  - SLoRA:    outputs/quick_gemma_4b"
echo ""
echo "Check W&B project 'slora' for metrics comparison"
