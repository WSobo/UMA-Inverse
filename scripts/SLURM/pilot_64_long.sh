#!/bin/bash
#SBATCH --job-name=umainv-p64L
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/pilot64L_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/pilot64L_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo "Extended-budget pilot (500 epochs, T_max=500, warmup=50) to test"
echo "the 'undertraining, not architecture' hypothesis."
echo "Target: train/loss < 0.3, train/acc > 0.85 by epoch 500"
uv run python -u scripts/pilot_run.py ++data.max_total_nodes=64 ++pilot.epochs=500
echo "Extended-pilot done. Check logs/pilot/pilot/version_*/metrics.csv."
