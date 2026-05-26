#!/bin/bash
#SBATCH --job-name=uma-inv-v4-stage1
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/v4_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/v4_train_%j.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2

export WANDB_MODE="offline"

# v4 STAGE 1 — single A100, max_total_nodes=64. Identical hyperparams to v3
# Stage 1 (15 epochs, bsz=8, warmup=1k, T_max=280k, all v3 feature flags ON).
# The decoder fix (learned ligand attention in _ligand_aware_context) is the
# only architectural change vs v3; all other knobs are held constant so the
# Stage 1 val/acc delta is a clean signal for the fix.
#
# Submit the full v4 curriculum hands-free with:
#   J1=$(sbatch --parsable scripts/SLURM/04a_v4_train_stage1.sh)
#   J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/SLURM/04b_v4_train_stage2_ddp.sh)
#   J3=$(sbatch --parsable --dependency=afterok:$J2 scripts/SLURM/04c_v4_train_stage3_ddp.sh)

echo ">> [v4 STAGE 1] 15 epochs at max_total_nodes=64, bsz=8, all v3 flags ON (A100)"
uv run python scripts/train.py \
    run_name="pairmixerinv-v4-stage1-nodes64" \
    ++trainer.max_epochs=15 \
    ++data.max_total_nodes=64 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.warmup_steps=1000 \
    ++training.T_max=280000 \
    ++data.pair_distance_atoms=backbone_full_25 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.ligand_featurizer=ligandmpnn_atomic \
    ++data.frame_relative_angles=true \
    ++data.return_sidechain_atoms=true \
    ++data.sidechain_context_rate=0.03 \
    ++training.coord_noise_std=0.1

echo "v4 Stage 1 complete. last.ckpt at checkpoints/pairmixerinv-v4-stage1-nodes64/"
