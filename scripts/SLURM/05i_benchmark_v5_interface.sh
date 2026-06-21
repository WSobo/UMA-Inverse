#!/bin/bash
#SBATCH --job-name=uma-bench-v5-iface
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/bench_v5iface_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/bench_v5iface_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# ── v5 INTERFACE RECOVERY — the LigandMPNN head-to-head for the preprint ──────
# LigandMPNN published protocol (Dauparas et al.): for each PDB, 10 autoregressive
# samples at T=0.1, random decoding order, recovery restricted to sidechain-
# interface residues (<=5 A of any nonprotein heavy atom), median across the 10
# samples -> one scalar per PDB; headline = mean of per-PDB medians. Baselines are
# LigandMPNN's published numbers (small_molecule 0.633, metal 0.775, nucleotide
# 0.505) and ProteinMPNN (0.505 / 0.406 / 0.471) — fixed, so v5 is directly
# comparable just by running this protocol on the same fixed test splits.
#
# v5 NOTE: unlike v3 (which had no nucleotide context), v5's NA routing lets us
# evaluate the NUCLEOTIDE split — a comparison v3 could not make.
#
# Runs all three splits sequentially in one job (one clean afterok dependency).
# config_v5.yaml matches the v5 featurization; the distogram aux head is dropped
# by the strict=False inference loader (verified: trunk loads 257/257).

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

# expandable_segments avoids the CUDA OOM fragmentation seen on the 24 GB A5500
# for large complexes at M=50 (a 790 MiB alloc failed with 681 MiB free).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-}"
if [[ -z "$CKPT" ]]; then
    if [[ -f checkpoints/uma-inverse-v5.ckpt ]]; then
        CKPT=checkpoints/uma-inverse-v5.ckpt
    else
        CKPT=checkpoints/pairmixerinv-v5-stage3-nodes384-ddp2/epoch_snapshots/epoch-11.ckpt
    fi
fi
CONFIG="${CONFIG:-configs/config_v5.yaml}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.1}"
CUTOFF="${CUTOFF:-5.0}"
OUT_DIR="${OUT_DIR:-outputs/benchmark/interface_recovery}"
RUN_PREFIX="${RUN_PREFIX:-v5}"

[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint not found: $CKPT" >&2; exit 1; }
echo ">> v5 interface recovery: ckpt=$CKPT  config=$CONFIG  samples=$NUM_SAMPLES  T=$TEMPERATURE  cutoff=${CUTOFF}A  ($(date))"

for cls in small_molecule metal nucleotide; do
    val_json="LigandMPNN/training/test_${cls}.json"
    pdb_dir="data/raw/pdb_archive/test_${cls}"
    run_name="${RUN_PREFIX}-test_${cls}"
    if [[ ! -d "$pdb_dir" ]]; then
        echo "!! pdb dir missing: $pdb_dir (run scripts/SLURM/01b_download_test_splits.sh first) — skipping $cls" >&2
        continue
    fi
    echo ">> [$cls] run=$run_name  ($(date))"
    uv run python scripts/benchmark_interface_recovery.py \
        --ckpt "$CKPT" \
        --config "$CONFIG" \
        --val-json "$val_json" \
        --pdb-dir "$pdb_dir" \
        --run-name "$run_name" \
        --out-dir "$OUT_DIR" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --cutoff "$CUTOFF" \
        --max-total-nodes 5000 \
        --seed 0
done

echo ">> summarizing vs LigandMPNN/ProteinMPNN published numbers  ($(date))"
uv run python scripts/summarize_test_benchmarks.py \
    --run-prefix "$RUN_PREFIX" \
    --out-dir "$OUT_DIR" || echo "(summarize step optional — inspect per-split summary.json directly)"

echo ">> v5 interface recovery complete: $OUT_DIR/${RUN_PREFIX}-test_*  ($(date))"
