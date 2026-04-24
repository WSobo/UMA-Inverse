#!/bin/bash
#SBATCH --job-name=fetch_pdb
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/fetch_pdb_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/fetch_pdb_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo "Running on hosts: $SLURM_NODELIST"
echo "Timestamp: $(date)"

echo "Scraping specific JSON IDs from LigandMPNN and downloading flat files safely..."
# 16 worker threads: ~2x the 8-thread default. Each HTTP fetch is I/O-bound,
# so 16 on 4 CPUs is comfortable; the script's built-in 2s/4s/6s backoff keeps
# us polite to files.rcsb.org when transient errors hit.
uv run python scripts/download_json_pdbs.py --workers 16

echo "Targeted JSON download complete at: $(date)"
