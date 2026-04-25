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
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

export WANDB_MODE="offline"

echo "Starting UMA-Inverse v2 Stage 1 (single-GPU)..."

# Per-stage LR schedule (warmup + cosine T_max) is sized to *that stage's*
# total step count: steps/epoch ≈ 145K samples / batch_size. T_max is set
# slightly above total steps so the final LR floor is small-but-nonzero.
#
# batch_size and gradient_checkpointing are tuned per stage so the pair
# tensor (O(N^2)) fits on a 24 GB A5500 while keeping kernel launch overhead
# negligible. Throughput probe (2026-04-18) at bsz=8, N=64, grad_ckpt=off
# achieved 6.46 it/s on A5500.
#
# This script runs ONLY stage 1. Stages 2 and 3 are DDP and live in
# 03b_train_stage2_ddp.sh (4-GPU) and 03c_train_stage3_ddp.sh (8-GPU).
# Submit the full curriculum hands-free with:
#   J1=$(sbatch --parsable scripts/SLURM/03_train_model.sh)
#   J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/SLURM/03b_train_stage2_ddp.sh)
#   J3=$(sbatch --parsable --dependency=afterok:$J2 scripts/SLURM/03c_train_stage3_ddp.sh)
# Output checkpoints land at checkpoints/<run_name>/last.ckpt; stages 2+3
# read the previous stage's scoped path via init_from_checkpoint.

# =========================================================================
# STAGE 1: AGGRESSIVE CROPPING (N=64)
# Steps/epoch ≈ 145K / 8 = 18_100  |  15 epochs ≈ 272K steps  |  ETA ~12h
# =========================================================================
echo ">> [STAGE 1] 15 epochs at max_total_nodes=64, bsz=8, grad_ckpt=off"
uv run python scripts/train.py \
    run_name="pairmixerinv-v2-stage1-nodes64" \
    ++trainer.max_epochs=15 \
    ++data.max_total_nodes=64 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.warmup_steps=1000 \
    ++training.T_max=280000

echo "Stage 1 complete. last.ckpt at checkpoints/pairmixerinv-v2-stage1-nodes64/"
