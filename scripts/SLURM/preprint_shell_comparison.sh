#!/bin/bash
#SBATCH --job-name=shell-comparison
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/shell_comparison_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/shell_comparison_%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=medium

# Compare UMA-Inverse v3 vs LigandMPNN recovery by distance-to-ligand shell.
# Depends on: preprint_benchmark_ligandmpnn.sh + preprint_benchmark_uma_ar.sh

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

uv run python scripts/paper/shell_recovery_comparison.py

echo ">> Shell comparison complete."
echo ">> outputs/preprint/shell_recovery_summary.md"
cat outputs/preprint/shell_recovery_summary.md
