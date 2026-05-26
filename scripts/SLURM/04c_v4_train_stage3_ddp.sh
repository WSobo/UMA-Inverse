#!/bin/bash
#SBATCH --job-name=uma-inv-v4-stage3-ddp4
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/v4_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/v4_train_%j.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2

export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set — add it to ~/.bashrc}"

# v4 STAGE 3 DDP4 — 4×A100, max_total_nodes=384. bsz=4/rank gives effective
# batch=16, matching v3 Stage 3 (8×A5500 bsz=2/rank=16). Gradient checkpointing
# on to keep per-GPU memory in bounds at nodes=384. Same schedule as v3 Stage 3
# (30 epochs, warmup=2k, T_max=280k). Canonical v4 ckpt = min val_loss snapshot.

echo ">> [v4 STAGE 3 DDP4] 30 epochs at max_total_nodes=384, bsz=4/rank, devices=4 (A100)"
srun uv run python scripts/train.py \
    run_name="pairmixerinv-v4-stage3-nodes384-ddp4" \
    ++wandb.enabled=true \
    ++trainer.max_epochs=30 \
    ++data.max_total_nodes=384 \
    ++data.batch_size=4 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=true \
    ++training.devices=4 \
    ++training.warmup_steps=2000 \
    ++training.T_max=280000 \
    ++trainer.init_from_checkpoint="checkpoints/pairmixerinv-v4-stage2-nodes128-ddp2/last.ckpt" \
    ++data.pair_distance_atoms=backbone_full_25 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.ligand_featurizer=ligandmpnn_atomic \
    ++data.frame_relative_angles=true \
    ++data.return_sidechain_atoms=true \
    ++data.sidechain_context_rate=0.03 \
    ++training.coord_noise_std=0.1

# Promote the lowest-val-loss top-K snapshot to checkpoints/uma-inverse-v4.ckpt
CKPT_DIR=checkpoints/pairmixerinv-v4-stage3-nodes384-ddp4
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
