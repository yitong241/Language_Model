#!/bin/bash
#SBATCH --job-name=act_gelu
#SBATCH --partition=gpu --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/act_gelu_%j.out
#SBATCH --error=logs/act_gelu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@comp.nus.edu.sg

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "GPU Information:"
nvidia-smi
echo "=========================================="

source "$HOME/transformer_proj_venv/bin/activate"

export MPLBACKEND=Agg
export MPLCONFIGDIR="$HOME/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

cd "$HOME/5242/Language_Model"

OUTPUT_DIR="act_gelu/runs/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $(pwd)/$OUTPUT_DIR"
echo "=========================================="

python -u train.py --model-dir act_gelu --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "End Time: $(date)"