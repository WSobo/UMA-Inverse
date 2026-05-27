#!/bin/bash
#SBATCH --job-name=uma-inv-v4-stage3-ddp2
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/v4_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/v4_train_%j.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2

export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set — add it to ~/.bashrc}"

# v4 STAGE 3 DDP2 — 2×A100 80GB, max_total_nodes=384. bsz=8/rank gives effective
# batch=16, same as original DDP4 plan (effective batch unchanged, dynamics identical).
# 80GB VRAM makes gradient checkpointing unnecessary. Same schedule as v3 Stage 3
# (30 epochs, warmup=2k, T_max=280k). Canonical v4 ckpt = min val_loss snapshot.

STAGE2_DIR=checkpoints/pairmixerinv-v4-stage2-nodes128-ddp2
STAGE2_BEST=$(ls "${STAGE2_DIR}"/uma-inverse-*-*.ckpt 2>/dev/null \
    | awk -F'-' '{print $NF, $0}' | sort -n | head -1 | cut -d' ' -f2-)
STAGE2_CKPT="${STAGE2_BEST:-${STAGE2_DIR}/last.ckpt}"
echo ">> Stage 3 warm-starting from: ${STAGE2_CKPT}"

echo ">> [v4 STAGE 3 DDP2] 30 epochs at max_total_nodes=384, bsz=8/rank, devices=2 (A100)"
srun uv run python scripts/train.py \
    run_name="pairmixerinv-v4-stage3-nodes384-ddp2" \
    ++wandb.enabled=true \
    ++trainer.max_epochs=30 \
    ++data.max_total_nodes=384 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.devices=2 \
    ++training.warmup_steps=2000 \
    ++training.T_max=280000 \
    ++trainer.init_from_checkpoint="${STAGE2_CKPT}" \
    ++data.pair_distance_atoms=backbone_full_25 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.ligand_featurizer=ligandmpnn_atomic \
    ++data.frame_relative_angles=true \
    ++data.return_sidechain_atoms=true \
    ++data.sidechain_context_rate=0.03 \
    ++training.coord_noise_std=0.1

# Promote the lowest-val-loss top-K snapshot to checkpoints/uma-inverse-v4.ckpt
CKPT_DIR=checkpoints/pairmixerinv-v4-stage3-nodes384-ddp2
BEST_NAME=$(ls "${CKPT_DIR}"/uma-inverse-*-*.ckpt 2>/dev/null \
    | xargs -n1 basename \
    | sort -t- -k4 -n \
    | head -1)
[[ -n "$BEST_NAME" ]] && BEST="${CKPT_DIR}/${BEST_NAME}" || BEST=""
if [[ -n "$BEST" ]]; then
    cp "$BEST" checkpoints/uma-inverse-v4.ckpt
    echo "v4 final: $BEST -> checkpoints/uma-inverse-v4.ckpt"
else
    echo "WARNING: no top-K snapshot found in $CKPT_DIR; set --ckpt explicitly downstream"
fi

echo "v4 Stage 3 DDP4 complete."
