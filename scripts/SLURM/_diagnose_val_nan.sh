#!/bin/bash
#SBATCH --job-name=val-nan
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/valnan_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/valnan_%j.err
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
uv run python scripts/_diagnose_val_nan.py
