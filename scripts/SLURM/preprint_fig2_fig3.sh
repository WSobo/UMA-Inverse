#!/bin/bash
#SBATCH --job-name=preprint-fig2-3
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/preprint_fig23_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/preprint_fig23_%j.err
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Regenerate fig2 and fig3 once all three v3 interface-recovery jobs finish.
# Submit with --dependency=afterok:33742951:33742952:33742953

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> fig2_benchmark_bars.py"
uv run python scripts/paper/figures/fig2_benchmark_bars.py

echo ">> fig3_violins.py"
uv run python scripts/paper/figures/fig3_violins.py

echo ">> Done."
ls outputs/preprint/figures/fig2_benchmark_bars.png outputs/preprint/figures/fig3_violins.png
