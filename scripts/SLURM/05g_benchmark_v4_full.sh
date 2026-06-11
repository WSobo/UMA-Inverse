#!/bin/bash
#SBATCH --job-name=uma-bench-v4-full
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/bench_v4full_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/bench_v4full_%j.err
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# ── Full v4 benchmark — the AR *sampled* recovery the baseline skipped ────────
# The 05d baseline ran --skip-temperature (teacher-forced only, 0.689 pooled).
# This adds the temperature/diversity sweep → autoregressive sampled recovery,
# the metric directly comparable to LigandMPNN (0.5389) and v3 (0.4637 at K=0,
# T=0.1). v4 = the preprint's proposed decoder fix; this tests whether it
# closed the AR recovery gap.
#
# Checkpoint = the frozen ep6 reference (best val-loss; the Run A/Control fork
# point), config_v3.yaml (matches v4 featurization; loads 257/257 keys).
#
# Scope kept tractable (AR sampling is L sequential forwards/sample):
#   n=1000, temps {0.0, 0.1, 0.2}, 3 samples/PDB, ablation skipped (the 2000-PDB
#   baseline already has v4's ligand ablation). ~5-7 h on an A5500.
# Override any of these via the env vars below.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2

BENCH_CKPT="${BENCH_CKPT:-checkpoints/runA_v4_stage3_ep6_init.ckpt}"
BENCH_CONFIG="${BENCH_CONFIG:-configs/config_v3.yaml}"
BENCH_RUN_NAME="${BENCH_RUN_NAME:-v4-ep6-full}"
BENCH_N="${BENCH_N:-1000}"
BENCH_SEED="${BENCH_SEED:-0}"
BENCH_TEMPS="${BENCH_TEMPS:-0.0,0.1,0.2}"
BENCH_SAMPLES="${BENCH_SAMPLES:-3}"

echo ">> v4 full benchmark: ckpt=$BENCH_CKPT  run=$BENCH_RUN_NAME  n=$BENCH_N  temps=$BENCH_TEMPS  ($(date))"

uv run uma-inverse benchmark \
    --ckpt "$BENCH_CKPT" \
    --config "$BENCH_CONFIG" \
    --val-json LigandMPNN/training/valid.json \
    --pdb-dir data/raw/pdb_archive \
    --out-dir outputs/benchmark \
    --run-name "$BENCH_RUN_NAME" \
    --n-pdbs "$BENCH_N" \
    --seed "$BENCH_SEED" \
    --temperatures "$BENCH_TEMPS" \
    --samples-per-pdb "$BENCH_SAMPLES" \
    --skip-ablation \
    --max-total-nodes 5000 \
    -v

echo ">> v4 full benchmark complete: outputs/benchmark/$BENCH_RUN_NAME (summary.md)  ($(date))"