#!/bin/bash
#SBATCH --job-name=uma-kl-ref
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/kl_ref_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/kl_ref_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# ── KL-by-shell reference runs (v3 + LigandMPNN) on the SAME 394-PDB set ──────
# Runs the OLD repo's distal_kl_shift.py (the preprint's exact script) so the
# v3 checkpoint executes against FAITHFUL mean-pool code. The new branch's
# unconditional lig_ctx_attn (v4 decoder fix) would init random against the
# 256-key v3 ckpt and corrupt its ligand conditioning — so v3 MUST run here.
#
# Inputs (shared with the v4 run) live in the NEW repo and are passed as
# absolute paths; the old repo's own data/raw archive was moved out, so we
# point --pdb-dir at the new archive. Same pdb_list.txt + max-residues=359 ⇒
# identical evaluated set as outputs/preprint/distal_kl/v4-init-ep6.
#
#   KL_MODE=v3     → UMA-Inverse v3 (mean-pool), --skip-ligandmpnn  (~15 min)
#   KL_MODE=lmpnn  → LigandMPNN reference,        --skip-uma         (~60-90 min,
#                    shells out to micromamba ligandmpnn_env per PDB)
#
#   sbatch --export=ALL,KL_MODE=v3    scripts/SLURM/05f_kl_v3_lmpnn.sh
#   sbatch --export=ALL,KL_MODE=lmpnn scripts/SLURM/05f_kl_v3_lmpnn.sh

set -e
NEW=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2
OLD=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

MODE="${KL_MODE:-v3}"
case "$MODE" in
  v3)    SKIP="--skip-ligandmpnn"; SUB="v3-394" ;;
  lmpnn) SKIP="--skip-uma";        SUB="lmpnn-394" ;;
  *) echo "bad KL_MODE=$MODE (want v3|lmpnn)"; exit 1 ;;
esac

PDB_LIST="$NEW/outputs/preprint/distal_kl/pdb_list.txt"
PDB_DIR="$NEW/data/raw/pdb_archive"
OUT_DIR="$NEW/outputs/preprint/distal_kl/$SUB"

echo ">> KL reference run: mode=$MODE  out=$OUT_DIR  ($(date))"
cd "$OLD"

uv run python scripts/paper/distal_kl_shift.py \
    $SKIP \
    --uma-ckpt checkpoints/uma-inverse-v3.ckpt \
    --config configs/config_v3.yaml \
    --pdb-list "$PDB_LIST" \
    --pdb-dir "$PDB_DIR" \
    --out-dir "$OUT_DIR" \
    --num-batches 10 \
    --seed 0 \
    --max-residues 359

echo ">> done ($MODE): $OUT_DIR/distal_kl_summary.csv  ($(date))"