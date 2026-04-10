#!/bin/bash
#SBATCH --job-name=umainv
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/pilot_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/pilot_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo "Running 1-Batch Sanity Check (Pilot Run) at max_total_nodes=64..."
uv run python scripts/pilot_run.py ++data.max_total_nodes=64

echo "Running 1-Batch Sanity Check (Pilot Run) at max_total_nodes=128..."
uv run python scripts/pilot_run.py ++data.max_total_nodes=128

echo "Running 1-Batch Sanity Check (Pilot Run) at max_total_nodes=384..."
uv run python scripts/pilot_run.py ++data.max_total_nodes=384

echo "All Pilot runs completed successfully. Pipeline is confirmed for all curriculum stages!"
