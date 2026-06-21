#!/bin/bash
#SBATCH --job-name=uma-bench-runA
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/bench_runA_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/bench_runA_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# ── Run A before/after benchmark wrapper (UMA-Inverse) ──────────────────────
# NOTE: corrected paths — the stock scripts/SLURM/05_benchmark.sh still cd's to
# the OLD repo (…/UMA-Inverse, not …/UMA-Inverse).
#
# --config configs/config_v3.yaml is REQUIRED: InferenceSession builds the model
# architecture from the config YAML (not the ckpt) and loads weights strict=False
# — only warns on mismatch. config_v3.yaml encodes the shared v3/v4 featurization
# (backbone_full_25, frame_relative_angles, ligandmpnn_atomic, cb anchor); the v4
# decoder fix is unconditional code, so the current model matches the ckpt.
# Verified: v4 init loads 257/257 keys, 0 missing / 0 unexpected.
#
# Same config works for Run A and the control checkpoints — their only extra keys
# are the training-only distogram_head, which the bare UMAInverse drops at load.
#
# Defaults = the "BEFORE" baseline (the frozen v4 stage-3 epoch-6 fork point).
# Reuse VERBATIM for the "AFTER" so flags/subset/seed match exactly, e.g.:
#   sbatch --export=ALL,\
#BENCH_CKPT=checkpoints/pairmixerinv-runA-distogram-warmstart/last.ckpt,\
#BENCH_RUN_NAME=runA-distogram-ep3 scripts/SLURM/05d_benchmark_runA.sh

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

BENCH_CKPT="${BENCH_CKPT:-checkpoints/runA_v4_stage3_ep6_init.ckpt}"
BENCH_CONFIG="${BENCH_CONFIG:-configs/config_v3.yaml}"
BENCH_VAL_JSON="${BENCH_VAL_JSON:-LigandMPNN/training/valid.json}"
BENCH_PDB_DIR="${BENCH_PDB_DIR:-data/raw/pdb_archive}"
BENCH_OUT_DIR="${BENCH_OUT_DIR:-outputs/benchmark}"
BENCH_RUN_NAME="${BENCH_RUN_NAME:-v4-init-ep6-baseline}"
BENCH_N="${BENCH_N:-2000}"
BENCH_SEED="${BENCH_SEED:-0}"

echo ">> benchmark: ckpt=$BENCH_CKPT  config=$BENCH_CONFIG  run=$BENCH_RUN_NAME  n=$BENCH_N  seed=$BENCH_SEED"

# Teacher-forced recovery (K=0) + ligand-context ablation. Temperature/diversity
# sweep skipped — not needed for the before/after distogram comparison and it is
# the slow (autoregressive sampling) pass. Re-enable by dropping --skip-temperature.
uv run uma-inverse benchmark \
    --ckpt "$BENCH_CKPT" \
    --config "$BENCH_CONFIG" \
    --val-json "$BENCH_VAL_JSON" \
    --pdb-dir "$BENCH_PDB_DIR" \
    --out-dir "$BENCH_OUT_DIR" \
    --run-name "$BENCH_RUN_NAME" \
    --n-pdbs "$BENCH_N" \
    --seed "$BENCH_SEED" \
    --skip-temperature \
    --max-total-nodes 5000 \
    -v

echo ">> benchmark complete: $BENCH_OUT_DIR/$BENCH_RUN_NAME (read summary.md)"
