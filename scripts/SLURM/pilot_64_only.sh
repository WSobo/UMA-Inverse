#!/bin/bash
#SBATCH --job-name=umainv-p64
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/pilot64_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/pilot64_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo "Pilot-64 verification run: multi-head AR attention fix"
echo "Target: train/loss < 0.3, train/acc > 0.85 by epoch 100"
uv run python -u scripts/pilot_run.py ++data.max_total_nodes=64
echo "Pilot-64 done. Check logs/pilot/pilot/version_*/metrics.csv for convergence."
