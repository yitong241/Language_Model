#!/bin/bash
#SBATCH --job-name=ffn_analysis
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
echo "=========================================="
echo "FFN Analysis Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
nvidia-smi
echo "=========================================="

# -----------------------------
# activate env
# -----------------------------
source $HOME/transformer_proj_venv/bin/activate

# -----------------------------
# paths (EDIT THESE)
# -----------------------------

BASE_DIR=$HOME/5242/Language_Model

RELU_CKPT=$BASE_DIR/baseline/runs/123456/baseline.pt
GELU_CKPT=$BASE_DIR/act_gelu/runs/522850/act_gelu.pt
SWIGLU_CKPT=$BASE_DIR/act_swiglu/runs/522698/act_swiglu.pt

UTILS_PATH=$BASE_DIR/utils.py

OUTPUT_DIR=$BASE_DIR/activation/ffn_analysis

mkdir -p $OUTPUT_DIR
mkdir -p logs

# -----------------------------
# run analysis
# -----------------------------

python act_gelu/analysis.py \
  --models \
    relu=$RELU_CKPT \
    gelu=$GELU_CKPT \
    swiglu=$SWIGLU_CKPT \
  --utils-py $UTILS_PATH \
  --output-dir $OUTPUT_DIR \
  --batch-size 16 \
  --num-batches 100 \
  --eval-num-tokens 500000 \
  --device cuda

echo "=========================================="
echo "Finished at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="