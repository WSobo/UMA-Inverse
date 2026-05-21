#!/bin/bash
#SBATCH --job-name=pf-metrics-v3
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/pf_metrics_v3_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/pf_metrics_v3_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Regenerate pocket_fixed_summary.csv, pocket_fixed_metrics.csv, and
# pocket_fixed_aa_freq.csv using v3 designs (uma_pocket_fixed_v3, 62 PDBs)
# paired with the combined2 selection (50 small_mol + 10 metal = 60 PDBs).
#
# Outputs (overwrites existing):
#   outputs/preprint/pocket_fixed_summary.csv   (method = uma_v3)
#   outputs/preprint/pocket_fixed_metrics.csv
#   outputs/preprint/pocket_fixed_aa_freq.csv

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

uv run python scripts/paper/compute_pocket_fixed_metrics.py \
    --uma-dir      outputs/preprint/uma_pocket_fixed_v3 \
    --uma-method   uma_v3 \
    --ligandmpnn-dir outputs/preprint/ligandmpnn_pocket_fixed \
    --selection    outputs/preprint/pdb_selection_combined2.json \
    --out-dir      outputs/preprint

echo ">> Done. pocket_fixed_summary.csv regenerated with uma_v3."
