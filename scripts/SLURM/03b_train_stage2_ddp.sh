#!/bin/bash
#SBATCH --job-name=uma-inv-stage2-ddp4
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

export WANDB_MODE="offline"

# Stage 2 DDP on 4x A5500 (single phoenix node).
# Loads stage-1/2 weights via init_from_checkpoint (weights-only). Optimizer
# and LR scheduler start fresh so warmup + cosine cycle are sized to THIS
# stage's 25-epoch budget, not the cumulative global_step from the loaded
# checkpoint. This is what fixes the LR=0 bug we hit on the prior attempt.
#
# Effective global batch = 4 (per-rank) x 4 (ranks) = 16.
# Steps/epoch = 145K samples / 16 ≈ 9_060. 25 epochs ≈ 227K steps.
# T_max=230K leaves a small LR floor at the end.
#
# `srun` forks 4 tasks so Lightning's SLURMEnvironment sees 4 ranks and
# binds each to one GPU. No `strategy="ddp"` in code — Lightning
# auto-selects it when devices>1 + accelerator=gpu.

echo ">> [STAGE 2 DDP4] 25 epochs at max_total_nodes=128, bsz=4/rank, devices=4"
srun uv run python scripts/train.py \
    run_name="pairmixerinv-v2-stage2-nodes128-ddp4" \
    ++trainer.max_epochs=25 \
    ++data.max_total_nodes=128 \
    ++data.batch_size=4 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.devices=4 \
    ++training.warmup_steps=1000 \
    ++training.T_max=230000 \
    ++trainer.init_from_checkpoint="checkpoints/pairmixerinv-v2-stage1-nodes64/last-v1.ckpt"

echo "Stage 2 DDP complete."
