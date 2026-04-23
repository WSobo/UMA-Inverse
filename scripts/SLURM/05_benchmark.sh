#!/bin/bash
#SBATCH --job-name=uma-benchmark
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/benchmark_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/benchmark_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Benchmark wrapper. Defaults to n_pdbs=500 for a fast dev pass (~30 min on
# A5500). Override with `--export=ALL,BENCH_N=all` (full ~7153 validation
# split) or `--export=ALL,BENCH_N=1000` for a mid-sized run.
#
# Usage:
#   sbatch scripts/SLURM/05_benchmark.sh                           # n=500
#   sbatch --export=ALL,BENCH_N=all scripts/SLURM/05_benchmark.sh
#   sbatch --export=ALL,BENCH_CKPT=path/to.ckpt scripts/SLURM/05_benchmark.sh

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

# ── Defaults — override via sbatch --export ──────────────────────────────────
BENCH_CKPT="${BENCH_CKPT:-checkpoints/last.ckpt}"
BENCH_VAL_JSON="${BENCH_VAL_JSON:-LigandMPNN/training/valid.json}"
BENCH_PDB_DIR="${BENCH_PDB_DIR:-data/raw/pdb_archive}"
BENCH_OUT_DIR="${BENCH_OUT_DIR:-outputs/benchmark}"
BENCH_RUN_NAME="${BENCH_RUN_NAME:-${SLURM_JOB_ID:-manual}}"
BENCH_N="${BENCH_N:-500}"

echo ">> benchmark config:"
echo "     ckpt       = $BENCH_CKPT"
echo "     val json   = $BENCH_VAL_JSON"
echo "     pdb dir    = $BENCH_PDB_DIR"
echo "     out dir    = $BENCH_OUT_DIR"
echo "     run name   = $BENCH_RUN_NAME"
echo "     n pdbs     = $BENCH_N"

N_FLAG=""
if [[ "$BENCH_N" == "all" ]]; then
    N_FLAG="--all"
else
    N_FLAG="--n-pdbs $BENCH_N"
fi

uv run uma-inverse benchmark \
    --ckpt "$BENCH_CKPT" \
    --val-json "$BENCH_VAL_JSON" \
    --pdb-dir "$BENCH_PDB_DIR" \
    --out-dir "$BENCH_OUT_DIR" \
    --run-name "$BENCH_RUN_NAME" \
    $N_FLAG \
    --max-total-nodes 5000 \
    -v

echo ">> benchmark complete."
