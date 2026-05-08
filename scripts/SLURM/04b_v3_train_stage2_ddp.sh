#!/bin/bash
#SBATCH --job-name=uma-inv-v3-stage2-ddp4
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/v3_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/v3_train_%j.err
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

# v3 STAGE 2 DDP4 — 4× A5500, max_total_nodes=128, all v3 flags ON.
# Same schedule as v2 stage 2 (25 epochs, bsz=4/rank, warmup=1k, T_max=230k).
# Loads stage-1 weights via init_from_checkpoint (weights-only).

echo ">> [v3 STAGE 2 DDP4] 25 epochs at max_total_nodes=128, bsz=4/rank, devices=4"
srun uv run python scripts/train.py \
    run_name="pairmixerinv-v3-stage2-nodes128-ddp4" \
    ++trainer.max_epochs=25 \
    ++data.max_total_nodes=128 \
    ++data.batch_size=4 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.devices=4 \
    ++training.warmup_steps=1000 \
    ++training.T_max=230000 \
    ++trainer.init_from_checkpoint="checkpoints/pairmixerinv-v3-stage1-nodes64/last.ckpt" \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.frame_relative_angles=true \
    ++data.sidechain_context_rate=0.03 \
    ++model.intra_ligand_multidist=true \
    ++training.coord_noise_std=0.1

echo "v3 Stage 2 DDP complete."
