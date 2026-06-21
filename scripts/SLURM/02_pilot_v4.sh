#!/bin/bash
#SBATCH --job-name=uma-inv-v4-pilot
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/v4_pilot_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/v4_pilot_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

# v4 3-stage pilot, 2×A100. Same convergence gate as v3 pilot:
# each stage must reach train/loss<0.3, train/acc>0.85.
# Stage 3 (nodes=384) is heavier per step; 300 epochs is enough to confirm
# the decoder fix doesn't break the overfit path.

echo ">> [v4 STAGE 1] 500-epoch pilot at max_total_nodes=64 (DDP2, A100)"
srun uv run python -u scripts/pilot_run.py ++data.max_total_nodes=64 ++pilot.epochs=500 ++training.devices=2

echo ">> [v4 STAGE 2] 500-epoch pilot at max_total_nodes=128 (DDP2, A100)"
srun uv run python -u scripts/pilot_run.py ++data.max_total_nodes=128 ++pilot.epochs=500 ++training.devices=2

echo ">> [v4 STAGE 3] 300-epoch pilot at max_total_nodes=384 (DDP2, A100)"
srun uv run python -u scripts/pilot_run.py ++data.max_total_nodes=384 ++pilot.epochs=300 ++training.devices=2

echo "All v4 pilot stages done. Gate: each stage reaches train/loss<0.3, train/acc>0.85."
