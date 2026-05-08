#!/bin/bash
#SBATCH --job-name=uma-inv-v3-stage1
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/v3_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/v3_train_%j.err
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

# v3 STAGE 1 — single-GPU, max_total_nodes=64, all 5 LigandMPNN-matched
# features ON. Same curriculum schedule as v2 stage 1 (15 epochs, bsz=8,
# warmup=1k, T_max=280k) so results are directly comparable.
#
# Submit the full v3 curriculum hands-free with:
#   J1=$(sbatch --parsable scripts/SLURM/04a_v3_train_stage1.sh)
#   J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/SLURM/04b_v3_train_stage2_ddp.sh)
#   J3=$(sbatch --parsable --dependency=afterok:$J2 scripts/SLURM/04c_v3_train_stage3_ddp.sh)

echo ">> [v3 STAGE 1] 15 epochs at max_total_nodes=64, bsz=8, all v3 flags ON"
uv run python scripts/train.py \
    run_name="pairmixerinv-v3-stage1-nodes64" \
    ++trainer.max_epochs=15 \
    ++data.max_total_nodes=64 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.warmup_steps=1000 \
    ++training.T_max=280000 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.frame_relative_angles=true \
    ++data.sidechain_context_rate=0.03 \
    ++model.intra_ligand_multidist=true \
    ++training.coord_noise_std=0.1

echo "v3 Stage 1 complete. last.ckpt at checkpoints/pairmixerinv-v3-stage1-nodes64/"
