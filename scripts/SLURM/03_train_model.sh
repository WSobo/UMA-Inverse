#!/bin/bash
#SBATCH --job-name=uma-traininv
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/train_%j.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
eval "$(micromamba shell hook --shell bash)"
micromamba activate uma-fold

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

# WANDB Setup
export WANDB_MODE="offline"

python -c "import torch; torch.set_float32_matmul_precision('high');"

echo "Starting Full UMA-Inverse Training Campaign (Curriculum Pipeline)..."

# =========================================================================
# STAGE 1: AGGRESSIVE CROPPING
# =========================================================================
echo ">> [STAGE 1] Running 15 Epochs on max_total_nodes=64"
python scripts/train.py \
    run_name="pairmixerinv-stage1-nodes64" \
    ++trainer.max_epochs=15 \
    ++data.max_total_nodes=64

# =========================================================================
# STAGE 2: INTERMEDIATE CROPPING
# =========================================================================
echo ">> [STAGE 2] Running up to Epoch 40 on max_total_nodes=128"
python scripts/train.py \
    run_name="pairmixerinv-stage2-nodes128" \
    ++trainer.max_epochs=40 \
    ++data.max_total_nodes=128 \
    ++trainer.resume_from_checkpoint="checkpoints/last.ckpt"

# =========================================================================
# STAGE 3: FULL CONTEXT
# =========================================================================
echo ">> [STAGE 3] Running up to Epoch 100 on max_total_nodes=384"
python scripts/train.py \
    run_name="pairmixerinv-stage3-nodes384" \
    ++trainer.max_epochs=100 \
    ++data.max_total_nodes=384 \
    ++trainer.resume_from_checkpoint="checkpoints/last.ckpt"

echo "Full Curriculum Campaign Complete!"
