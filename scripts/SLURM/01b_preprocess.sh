#!/bin/bash
#SBATCH --job-name=uma-preproc
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/uma-preproc_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/uma-preproc_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=medium
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
eval "$(micromamba shell hook --shell bash)"
micromamba activate uma-fold

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo "Running on node: $SLURM_NODELIST"
echo "Timestamp: $(date)"

echo "Kicking off CPU-only preprocessing to cache .pt files..."
python scripts/preprocess.py

echo "Pre-processing completed at: $(date)"
