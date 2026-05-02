#!/bin/bash
#SBATCH --job-name=preprint-finalize
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/finalize_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/finalize_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Phase A scale-up final step: re-run all pocket-fixed metric/stat/figure
# scripts on the combined N=25 small_mol + 10 metal selection. CPU-only.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

uv run python scripts/preprint/finalize_combined_metrics.py "$@"

echo "Done."
