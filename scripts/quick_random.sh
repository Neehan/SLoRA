#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -A mit_general
#SBATCH --job-name=quick_random
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=64GB
#SBATCH -t 48:00:00

module load miniforge/24.3.0-0

set -e

echo "=== Random Baseline Test ==="
echo "Starting time: $(date)"

PROJECT_ROOT="/home/notadib/projects/SLoRA"
cd ${PROJECT_ROOT}

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export WANDB_PROJECT="flora"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export TRANSFORMERS_ATTENTION_BACKEND=SDPA
export FLASH_ATTENTION_SKIP=True

mkdir -p logs

echo "Running Random baseline..."
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/train.py \
    --config configs/random_gemma3_1b_it.yaml 2>&1 | tee logs/random_gemma3_1b_it.log

echo "Random baseline complete!"
echo "End time: $(date)"
echo ""
echo "Results: outputs/random_gemma3_1b_it"
echo "Check W&B project 'flora' for metrics"
