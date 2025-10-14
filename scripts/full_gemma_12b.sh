#!/bin/bash
#SBATCH --job-name=slora_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=96:00:00
#SBATCH --mem=512G

set -e

echo "=== SLoRA Full Experiment (Gemma-3-12B-it, 400k samples) ==="
echo "Starting time: $(date)"

cd "$(dirname "$0")/.."

source ~/.bashrc
conda activate slora

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="slora"

echo "Starting full SLoRA run..."
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_slora.py \
    --config configs/full_gemma3_12b_it.yaml

echo "Full experiment complete!"
echo "End time: $(date)"
echo "Results: outputs/full_gemma3_12b_it"
