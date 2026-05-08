#!/bin/bash
#SBATCH --job-name=uma-inv-v3-pilot
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/v3_pilot_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/v3_pilot_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

export WANDB_MODE="offline"

# v3 pilot smoke test — ALL 5 v3 flags ON, single GPU, 1-batch overfit at
# max_total_nodes=64. Verifies that the new tensors plumb through forward
# without NaN before launching the full curriculum (~3-4 weeks compute).
# Gate: train/loss decreases monotonically; val pass executes; train/acc>0.85.

echo ">> [v3 PILOT] 1-GPU, max_total_nodes=64, all v3 flags ON"
srun uv run python -u scripts/pilot_run.py \
    ++data.max_total_nodes=64 \
    ++pilot.epochs=300 \
    ++training.devices=1 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.frame_relative_angles=true \
    ++data.sidechain_context_rate=0.03 \
    ++model.intra_ligand_multidist=true \
    ++training.coord_noise_std=0.1

echo "v3 pilot done."
