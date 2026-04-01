#!/bin/bash
#SBATCH --job-name=pe_learned
#SBATCH --partition=gpu --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/pe_learned_%j.out
#SBATCH --error=logs/pe_learned_%j.err
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

if conda env list | grep -q "^transformer_proj "; then
    echo "Conda environment 'transformer_proj' already exists. Activating it..."
    conda activate transformer_proj
else
    echo "Creating new conda environment 'transformer_proj'..."
    conda create -n transformer_proj python=3.10 -y
    conda activate transformer_proj
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio
    python -m pip install kagglehub datasets transformers
fi

cd $HOME/project/Language_Model

OUTPUT_DIR="pe_alibi/runs/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $(pwd)/$OUTPUT_DIR"
echo "=========================================="

python -u train.py --model-dir pe_alibi --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "End Time: $(date)"