#!/bin/bash
#SBATCH --job-name=preprint-analysis
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/preprint_analysis_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/preprint_analysis_%j.err
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Run distal signal analysis and paired stats on the regenerated v3 pocket-fixed data.
# Submit after preprint_pocket_fixed_metrics_v3.sh completes.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> distal_signal_analysis.py"
uv run python scripts/paper/distal_signal_analysis.py \
    --summary outputs/preprint/pocket_fixed_summary.csv \
    --aa-freq  outputs/preprint/pocket_fixed_aa_freq.csv \
    --cofold   outputs/preprint/cofold_metrics_ext2.csv

echo ">> pocket_fixed_stats.py"
uv run python scripts/paper/pocket_fixed_stats.py \
    --summary outputs/preprint/pocket_fixed_summary.csv \
    --out      outputs/preprint/pocket_fixed_stats.txt

echo ">> Done. Stats written to outputs/preprint/pocket_fixed_stats.txt"
