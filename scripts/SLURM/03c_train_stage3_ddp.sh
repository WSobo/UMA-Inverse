#!/bin/bash
#SBATCH --job-name=uma-inv-stage3-ddp8
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/train_%j.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

export WANDB_MODE="offline"

# Stage 3 DDP on 8x A5500 (one full phoenix node).
# Loads stage-2 weights via init_from_checkpoint pointing at stage 2's scoped
# checkpoint dir (no manual cp required — every run writes to its own subdir
# of checkpoints/ now). Fresh optimizer + LR scheduler so warmup + cosine are
# sized to THIS stage's 30-epoch budget.
#
# Effective global batch = 2 (per-rank) x 8 (ranks) = 16.
# Steps/epoch = 145K samples / 16 ≈ 9_060. 30 epochs ≈ 272K steps.
# T_max=280K rounds up slightly for the LR floor.
#
# Epoch count (30): stage 1 converged L=0.003 in 15 epochs on the pilot;
# 30 full-data epochs at max context is still >2× stage 1's budget and
# keeps wall-clock in the 10–18 day range.
#
# gradient_checkpointing=true kept here — N=384 pair tensor × 6 blocks
# at bsz=2 is still tight on 24 GB without it.

echo ">> [STAGE 3 DDP8] 30 epochs at max_total_nodes=384, bsz=2/rank, devices=8"
srun uv run python scripts/train.py \
    run_name="pairmixerinv-v2-stage3-nodes384-ddp8" \
    ++trainer.max_epochs=30 \
    ++data.max_total_nodes=384 \
    ++data.batch_size=2 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=true \
    ++training.devices=8 \
    ++training.warmup_steps=2000 \
    ++training.T_max=280000 \
    ++trainer.init_from_checkpoint="checkpoints/pairmixerinv-v2-stage2-nodes128-ddp4/last.ckpt"

echo "Stage 3 DDP complete."
