#!/bin/bash
#SBATCH --job-name=act_swiglu
#SBATCH --partition=gpu --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/act_swiglu_%j.out
#SBATCH --error=logs/act_swiglu_%j.err
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

# ---- Activate or create virtual environment ----
if [ -d "$HOME/transformer_proj_venv" ]; then
    echo "Using existing virtual environment..."
else
    echo "Creating virtual environment..."
    python3 -m venv "$HOME/transformer_proj_venv"
fi

source "$HOME/transformer_proj_venv/bin/activate"

# ---- Ensure required packages are installed ----
python -m pip install --upgrade pip
pip install --quiet torch datasets transformers kagglehub matplotlib pandas pyarrow tqdm

# ---- Go to project root (IMPORTANT FIX) ----
cd "$HOME/5242/Language_Model"

# ---- Output directory ----
OUTPUT_DIR="act_swiglu/runs/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $(pwd)/$OUTPUT_DIR"
echo "=========================================="

# ---- Start training ----
python -u train.py --model-dir act_swiglu --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "End Time: $(date)"