#!/bin/bash
#SBATCH --job-name=uma-smoke
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/smoke_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# End-to-end inference smoke test: loads the stage-3 ep11 checkpoint,
# encodes a fixture PDB (1bc8), runs teacher-forced scoring + AR
# sampling, prints recovery overall and within 5 Å of the ligand.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> smoke test on host $SLURM_NODELIST at $(date)"
uv run python notebooks/inference_smoke_test.py
echo ">> smoke test finished at $(date)"
