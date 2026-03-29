#!/bin/bash
#SBATCH --job-name=norm_post_ln
#SBATCH --partition=gpu --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/norm_post_ln_%j.out
#SBATCH --error=logs/norm_post_ln_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@comp.nus.edu.sg

mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "GPU Information:"
nvidia-smi
echo "=========================================="

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda installation not found!"
    exit 1
fi

if conda env list | grep -q "^transformer_proj"; then
    echo "Conda environment 'transformer_proj' already exists. Activating it..."
    conda activate transformer_proj
else
    echo "Creating new conda environment 'transformer_proj'..."
    conda env create -f $HOME/env.yml
    conda activate transformer_proj
    pip install --upgrade pip
    pip install kagglehub
    pip install datasets
    pip install transformers
fi

cd $HOME/Language_Model

OUTPUT_DIR="norm_post_ln/runs/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $(pwd)/$OUTPUT_DIR"
echo "=========================================="

python -u train.py --model-dir norm_post_ln --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "End Time: $(date)"
