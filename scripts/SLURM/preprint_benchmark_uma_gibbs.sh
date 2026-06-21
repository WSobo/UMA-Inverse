#!/bin/bash
#SBATCH --job-name=uma-gibbs
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/benchmark_gibbs_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/benchmark_gibbs_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# UMA-Inverse v3 Gibbs sampling sweep on 2000 val PDBs.
#
# Sweeps over num_iterations in {0, 1, 2, 3, 5}:
#   K=0  structure-only (no sequence context) — baseline
#   K=1  one full bidirectional pass
#   K=3  near-convergence
#   K=5  converged
#
# Each iteration is ONE forward pass (not L), so this is ~40x cheaper
# per sample than the sequential AR sweep.
#
# Output: outputs/benchmark/v3-gibbs-K5/gibbs_sweep.csv

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

uv run uma-inverse benchmark \
    --ckpt              checkpoints/uma-inverse-v3.ckpt \
    --config            configs/old_configs/config_v3.yaml \
    --val-json          LigandMPNN/training/valid.json \
    --pdb-dir           data/raw/pdb_archive \
    --out-dir           outputs/benchmark \
    --run-name          v3-gibbs-K5 \
    --n-pdbs            2000 \
    --skip-ablation \
    --skip-temperature \
    --no-skip-gibbs \
    --gibbs-iterations  0,1,2,3,5 \
    --samples-per-pdb   3 \
    --max-total-nodes   650 \
    -v

echo ">> Gibbs sweep complete."
echo ">> Results: outputs/benchmark/v3-gibbs-K5/gibbs_sweep.csv"
cat outputs/benchmark/v3-gibbs-K5/gibbs_sweep.csv
