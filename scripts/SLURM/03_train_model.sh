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

echo "Starting Full UMA-Inverse Training Campaign (Curriculum Pipeline)..."

# Per-stage LR schedule (warmup + cosine T_max) is sized to *that stage's*
# total step count: steps/epoch ≈ 145K samples / batch_size. T_max is set
# slightly above total steps so the final LR floor is small-but-nonzero.
#
# batch_size and gradient_checkpointing are tuned per stage so the pair
# tensor (O(N^2)) fits on a 24 GB A5500 while keeping kernel launch overhead
# negligible. Throughput probe (2026-04-18) at bsz=8, N=64, grad_ckpt=off
# achieved 6.46 it/s on A5500.

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

# =========================================================================
# STAGE 2: INTERMEDIATE CROPPING (N=128)
# Steps/epoch ≈ 145K / 4 = 36_300  |  25 delta epochs ≈ 908K steps
# =========================================================================
echo ">> [STAGE 2] 25 epochs at max_total_nodes=128, bsz=4"
uv run python scripts/train.py \
    run_name="pairmixerinv-v2-stage2-nodes128" \
    ++trainer.max_epochs=25 \
    ++data.max_total_nodes=128 \
    ++data.batch_size=4 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.warmup_steps=2000 \
    ++training.T_max=910000 \
    ++trainer.init_from_checkpoint="checkpoints/pairmixerinv-v2-stage1-nodes64/last.ckpt"

# =========================================================================
# STAGE 3: FULL CONTEXT (N=384)
# Steps/epoch ≈ 145K / 2 = 72_500  |  60 delta epochs ≈ 4.35M steps
# grad_ckpt=true reintroduced here because 384² pair tensor activations
# across 6 blocks approach 24GB at bsz=2 without checkpointing.
# =========================================================================
echo ">> [STAGE 3] 60 epochs at max_total_nodes=384, bsz=2, grad_ckpt=on"
uv run python scripts/train.py \
    run_name="pairmixerinv-v2-stage3-nodes384" \
    ++trainer.max_epochs=60 \
    ++data.max_total_nodes=384 \
    ++data.batch_size=2 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=true \
    ++training.warmup_steps=5000 \
    ++training.T_max=4400000 \
    ++trainer.init_from_checkpoint="checkpoints/pairmixerinv-v2-stage2-nodes128/last.ckpt"

echo "Full Curriculum Campaign Complete!"
