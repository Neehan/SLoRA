#!/bin/bash
#SBATCH --job-name=flora_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=96:00:00
#SBATCH --mem=512G

set -e

echo "=== FLoRA Full Experiment (Gemma-3-12B-PT, 400k samples) ==="
echo "Starting time: $(date)"

cd "$(dirname "$0")/.."

source ~/.bashrc
conda activate flora

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="flora"

echo "Starting full FLoRA run..."
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py \
    --config configs/full_gemma3_12b_pt.yaml

echo "Full experiment complete!"
echo "End time: $(date)"
echo "Results: outputs/full_gemma3_12b_pt"
