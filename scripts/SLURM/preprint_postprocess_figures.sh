#!/bin/bash
#SBATCH --job-name=preprint-figures
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/preprint_figures_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/preprint_figures_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Final preprint figure generation + K=0 vs LigandMPNN comparison.
# Submit with --dependency=afterok:<lmpnn-matched job id>
# (lmpnn-matched job must complete before this runs)
#
# Depends on:
#   outputs/preprint/pocket_fixed_summary.csv    (uma_v3 rows, from pf-metrics-v3 job)
#   outputs/preprint/cofold_metrics_ext2.csv     (already exists)
#   outputs/benchmark/v3-gibbs-K0/gibbs_per_pdb.csv  (from uma-k0 job)
#   outputs/benchmark/ligandmpnn-matched-k0/per_pdb.csv  (from lmpnn-matched job)

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> compare_k0_ligandmpnn.py"
uv run python scripts/paper/compare_k0_ligandmpnn.py \
    --uma-k0    outputs/benchmark/v3-gibbs-K0/gibbs_per_pdb.csv \
    --ligandmpnn outputs/benchmark/ligandmpnn-matched-k0/per_pdb.csv

echo ">> fig4_training.py"
uv run python scripts/paper/figures/fig4_training.py

echo ">> fig5_pocket_distal.py"
uv run python scripts/paper/figures/fig5_pocket_distal.py

echo ">> fig6_cofold.py"
uv run python scripts/paper/figures/fig6_cofold.py

echo ">> fig7_distal_signal.py"
uv run python scripts/paper/figures/fig7_distal_signal.py

echo ""
echo ">> All figures written to outputs/preprint/figures/"
ls outputs/preprint/figures/*.png
