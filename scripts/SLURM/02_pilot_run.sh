#!/bin/bash
#SBATCH --job-name=umainv-pall
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/pilot_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/pilot_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

# 3-stage pilot-all verification. 500 epochs for stages 1/2 (fast); 300 for
# stage 3 (nodes=384, triangle-mul FLOPs scale as N^3 so each epoch is ~200x
# slower than stage 1). pilot_run.py scales warmup as pilot_epochs // 10.

echo ">> [STAGE 1] 500-epoch pilot at max_total_nodes=64"
uv run python -u scripts/pilot_run.py ++data.max_total_nodes=64 ++pilot.epochs=500

echo ">> [STAGE 2] 500-epoch pilot at max_total_nodes=128"
uv run python -u scripts/pilot_run.py ++data.max_total_nodes=128 ++pilot.epochs=500

echo ">> [STAGE 3] 300-epoch pilot at max_total_nodes=384"
uv run python -u scripts/pilot_run.py ++data.max_total_nodes=384 ++pilot.epochs=300

echo "All pilot stages done. Gate: each stage reaches train/loss<0.3, train/acc>0.85."
