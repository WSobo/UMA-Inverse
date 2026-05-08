#!/bin/bash
#SBATCH --job-name=uma-inv-v3-stage3-ddp8
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/v3_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/v3_train_%j.err
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

# v3 STAGE 3 DDP8 — 8× A5500 (one full phoenix node), max_total_nodes=384,
# all v3 flags ON. Same schedule as v2 stage 3 (30 epochs, bsz=2/rank,
# warmup=2k, T_max=280k). Canonical v3 ckpt = stage-3 epoch with min val_loss.

echo ">> [v3 STAGE 3 DDP8] 30 epochs at max_total_nodes=384, bsz=2/rank, devices=8"
srun uv run python scripts/train.py \
    run_name="pairmixerinv-v3-stage3-nodes384-ddp8" \
    ++trainer.max_epochs=30 \
    ++data.max_total_nodes=384 \
    ++data.batch_size=2 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=true \
    ++training.devices=8 \
    ++training.warmup_steps=2000 \
    ++training.T_max=280000 \
    ++trainer.init_from_checkpoint="checkpoints/pairmixerinv-v3-stage2-nodes128-ddp4/last.ckpt" \
    ++data.pair_distance_atoms=backbone_full_25 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.ligand_featurizer=ligandmpnn_atomic \
    ++data.frame_relative_angles=true \
    ++data.return_sidechain_atoms=true \
    ++data.sidechain_context_rate=0.03 \
    ++training.coord_noise_std=0.1

echo "v3 Stage 3 DDP complete."
