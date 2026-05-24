#!/bin/bash
#SBATCH --job-name=uma-benchmark-ar
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/benchmark_ar_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/benchmark_ar_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# UMA-Inverse v3 autoregressive evaluation at T=0.1 on the same 2000 val
# PDBs used by the teacher-forced benchmark (v3-final).
#
# Produces outputs/benchmark/v3-ar-T0.1/temperature_sweep.csv with
# mean_recovery comparable to LigandMPNN's T=0.1 autoregressive numbers.
#
# Skip Pass 1 (teacher-forced) and the ligand ablation — we already have
# those from v3-final.  Only run the temperature sweep at T=0.1.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

uv run uma-inverse benchmark \
    --ckpt              checkpoints/uma-inverse-v3.ckpt \
    --config            configs/config_v3.yaml \
    --val-json          LigandMPNN/training/valid.json \
    --pdb-dir           data/raw/pdb_archive \
    --out-dir           outputs/benchmark \
    --run-name          v3-ar-T0.1 \
    --n-pdbs            2000 \
    --skip-ablation \
    --temperatures      0.1 \
    --samples-per-pdb   3 \
    --max-total-nodes   5000 \
    -v

echo ">> AR benchmark complete."
echo ">> Results: outputs/benchmark/v3-ar-T0.1/temperature_sweep.csv"
