#!/bin/bash
#SBATCH --job-name=uma-bench-v5-full
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/bench_v5full_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/bench_v5full_%j.err
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# ── Full v5 benchmark — AR *sampled* recovery + temperature/diversity sweep ───
# Mirrors 05g (v4 full) for the v5 model: the metric directly comparable to
# LigandMPNN (0.5389), v3 (0.4637), and v4. config_v5.yaml matches the v5
# Stage-3 featurization (ligandmpnn_atomic, backbone_full_25, frame angles,
# sidechain atoms, M=50); the trunk loads fully and the training-only distogram
# aux head is dropped by the strict=False inference loader.
#
# Checkpoint resolution (override with BENCH_CKPT=...):
#   * checkpoints/uma-inverse-v5.ckpt if it exists (the min-val_loss snapshot
#     promoted at training-job exit), else
#   * the permanent epoch-11 snapshot (current best val_loss; plateaued model).
# Chain after training with:  sbatch --dependency=afterok:<resume_jobid> ...
#
# Scope (AR sampling is L sequential forwards/sample): n=1000, temps
# {0.0,0.1,0.2}, 3 samples/PDB, ablation skipped. ~5-7 h on an A5500.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

# expandable_segments avoids CUDA OOM fragmentation on the 24 GB A5500 for large
# complexes at M=50 (seen in the n=5 smoke test).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BENCH_CKPT="${BENCH_CKPT:-}"
if [[ -z "$BENCH_CKPT" ]]; then
    if [[ -f checkpoints/uma-inverse-v5.ckpt ]]; then
        BENCH_CKPT=checkpoints/uma-inverse-v5.ckpt
    else
        BENCH_CKPT=checkpoints/pairmixerinv-v5-stage3-nodes384-ddp2/epoch_snapshots/epoch-11.ckpt
    fi
fi
BENCH_CONFIG="${BENCH_CONFIG:-configs/config.yaml}"
BENCH_RUN_NAME="${BENCH_RUN_NAME:-v5-best-full}"
BENCH_N="${BENCH_N:-1000}"
BENCH_SEED="${BENCH_SEED:-0}"
BENCH_TEMPS="${BENCH_TEMPS:-0.0,0.1,0.2}"
BENCH_SAMPLES="${BENCH_SAMPLES:-3}"
# Ligand ablation (re-eval with ligand features zeroed -> mean recovery delta).
# Default skips it (faster); set BENCH_SKIP_ABLATION=0 to include it.
BENCH_SKIP_ABLATION="${BENCH_SKIP_ABLATION:-1}"
ABLATION_FLAG=""; [[ "$BENCH_SKIP_ABLATION" == "1" ]] && ABLATION_FLAG="--skip-ablation"

[[ -f "$BENCH_CKPT" ]] || { echo "FATAL: checkpoint not found: $BENCH_CKPT" >&2; exit 1; }
echo ">> v5 full benchmark: ckpt=$BENCH_CKPT  config=$BENCH_CONFIG  run=$BENCH_RUN_NAME  n=$BENCH_N  temps=$BENCH_TEMPS  ($(date))"

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
    $ABLATION_FLAG \
    --max-total-nodes 5000 \
    -v

echo ">> v5 full benchmark complete: outputs/benchmark/$BENCH_RUN_NAME (summary.md)  ($(date))"
