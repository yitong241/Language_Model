#!/bin/bash
#SBATCH --job-name=eval_pe_learned
#SBATCH --partition=gpu --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Usage (run from pe_learned/):
#   sbatch eval.sh runs/<jobid>/best_model.pt
#   sbatch eval.sh runs/<jobid>/best_model.pt "Your custom prompt here"

CHECKPOINT=${1:?Usage: sbatch eval.sh <checkpoint_path> [prompt]}
PROMPT=${2:-}

mkdir -p logs

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda installation not found!"
    exit 1
fi

conda activate transformer_proj

cd $HOME/Language_Model

if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="pe_learned/$CHECKPOINT"
fi

echo "Checkpoint: $CHECKPOINT"

if [ -n "$PROMPT" ]; then
    python -u eval.py --model-dir pe_learned --checkpoint "$CHECKPOINT" --prompt "$PROMPT"
else
    python -u eval.py --model-dir pe_learned --checkpoint "$CHECKPOINT"
fi
