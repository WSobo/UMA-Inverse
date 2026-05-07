#!/bin/bash
#SBATCH --job-name=fig7-distal-signal
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/fig7_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/fig7_%j.err
#SBATCH --time=0:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=short

# Generate fig 7: distal-confidence vs Boltz-2 cofold quality scatter,
# UMA-v2 vs LigandMPNN, small-molecule split, pocket-fixed redesign.
# CPU-only matplotlib render.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

uv run python scripts/paper/figures/fig7_distal_signal.py "$@"

echo "Done."
