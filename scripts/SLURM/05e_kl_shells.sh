#!/bin/bash
#SBATCH --job-name=uma-kl-shells
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/kl_shells_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/kl_shells_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# ── KL-by-distance-shell (ligand information) for ONE checkpoint ──────────────
# Reuses the preprint mechanism analysis: per-position KL(P_ligand || P_no-lig)
# over the 20-AA distribution (10 decoding orders, ligand-on vs ligand-zeroed),
# binned into shells 0-5 / 5-10 / 10-15 / 15-25 / >25 A.
#
# This is the metric that captures ligand information even where the argmax
# doesn't flip — the right way to compare Run A (distogram λ=0.2) vs Control
# (λ=0) vs the v4-init baseline, instead of the dilution-prone Δrecovery.
#
# --config configs/config_v3.yaml: InferenceSession builds the bare UMAInverse
# from the config YAML and loads strict=False. config_v3.yaml matches the v4
# featurization; Run A/Control's extra training-only distogram_head keys drop
# cleanly as "unexpected" (the head lives on the LightningModule, not here).
#
# Same shared pdb_list.txt across all three checkpoints → matched A/B set.
#
# Defaults = the "BEFORE" v4-init baseline. Reuse VERBATIM for Run A / Control:
#   sbatch --dependency=afterok:<runA_jobid> --export=ALL,\
# KL_CKPT=checkpoints/pairmixerinv-runA-distogram-warmstart/last.ckpt,\
# KL_RUN_NAME=runA-distogram-ep3 scripts/SLURM/05e_kl_shells.sh

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

KL_CKPT="${KL_CKPT:-checkpoints/runA_v4_stage3_ep6_init.ckpt}"
KL_CONFIG="${KL_CONFIG:-configs/config_v3.yaml}"
KL_PDB_LIST="${KL_PDB_LIST:-outputs/preprint/distal_kl/pdb_list.txt}"
KL_PDB_DIR="${KL_PDB_DIR:-data/raw/pdb_archive}"
KL_RUN_NAME="${KL_RUN_NAME:-v4-init-ep6}"
KL_NUM_BATCHES="${KL_NUM_BATCHES:-10}"
KL_SEED="${KL_SEED:-0}"
KL_MAX_RESIDUES="${KL_MAX_RESIDUES:-359}"
KL_OUT_DIR="${KL_OUT_DIR:-outputs/preprint/distal_kl/${KL_RUN_NAME}}"

echo ">> KL-shells: ckpt=$KL_CKPT  config=$KL_CONFIG  run=$KL_RUN_NAME"
echo "             pdb_list=$KL_PDB_LIST  num_batches=$KL_NUM_BATCHES  seed=$KL_SEED"

uv run python scripts/paper/distal_kl_shift.py \
    --skip-ligandmpnn \
    --uma-ckpt "$KL_CKPT" \
    --config "$KL_CONFIG" \
    --pdb-list "$KL_PDB_LIST" \
    --pdb-dir "$KL_PDB_DIR" \
    --out-dir "$KL_OUT_DIR" \
    --num-batches "$KL_NUM_BATCHES" \
    --seed "$KL_SEED" \
    --max-residues "$KL_MAX_RESIDUES"

echo ">> KL-shells complete: $KL_OUT_DIR (read distal_kl_summary.csv)"
