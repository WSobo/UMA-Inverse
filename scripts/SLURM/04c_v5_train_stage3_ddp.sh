#!/bin/bash
#SBATCH --job-name=uma-inv-v5-stage3-ddp2
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/v5_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/v5_train_%j.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2

export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set — add it to ~/.bashrc}"

# v5 STAGE 3 DDP2 — 2×A100, max_total_nodes=384, bsz=8/rank → effective batch
# 16 (the original design; schedule = 24 epochs, warmup=2k, T_max=280k).
# Gradient checkpointing ON (N=384 pair-tensor activations). num_workers=4/rank
# to keep Ceph I/O light; DDP collective timeout raised to 2h in train.py.
# Canonical v5 ckpt = min val_loss.

STAGE2_DIR=checkpoints/pairmixerinv-v5-stage2-nodes128-ddp2
STAGE2_BEST=$(ls "${STAGE2_DIR}"/uma-inverse-*-*.ckpt 2>/dev/null \
    | grep -Ev -- '-v[0-9]+\.ckpt$' \
    | awk -F'-' '{print $NF, $0}' | sort -n | head -1 | cut -d' ' -f2-)
STAGE2_CKPT="${STAGE2_BEST:-${STAGE2_DIR}/last.ckpt}"
echo ">> Stage 3 warm-starting from: ${STAGE2_CKPT}"

echo ">> [v5 STAGE 3 DDP2] 24 epochs at max_total_nodes=384, bsz=8/rank, devices=2 (A100)"
srun uv run python scripts/train.py \
    run_name="pairmixerinv-v5-stage3-nodes384-ddp2" \
    ++wandb.enabled=true \
    ++trainer.max_epochs=24 \
    ++data.max_total_nodes=384 \
    ++data.batch_size=8 \
    ++data.num_workers=4 \
    ++model.gradient_checkpointing=true \
    ++training.devices=2 \
    ++training.warmup_steps=2000 \
    ++training.T_max=280000 \
    ++trainer.init_from_checkpoint="${STAGE2_CKPT}" \
    ++paths.processed_dir=data/processed_v5 \
    ++data.pair_distance_atoms=backbone_full_25 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.ligand_featurizer=ligandmpnn_atomic \
    ++data.frame_relative_angles=true \
    ++data.return_sidechain_atoms=true \
    ++data.sidechain_context_rate=0.03 \
    ++training.coord_noise_std=0.1 \
    ++data.ligand_context_atoms=50 \
    ++data.ligand_rich_features=false \
    ++data.ligand_bond_topology=false \
    ++model.distogram_aux_weight=0.2 \
    ++model.distogram_num_bins=38

# Promote the lowest-val-loss top-K snapshot to checkpoints/uma-inverse-v5.ckpt
CKPT_DIR=checkpoints/pairmixerinv-v5-stage3-nodes384-ddp2
BEST_NAME=$(ls "${CKPT_DIR}"/uma-inverse-*-*.ckpt 2>/dev/null \
    | xargs -n1 basename \
    | grep -Ev -- '-v[0-9]+\.ckpt$' \
    | sort -t- -k4 -n \
    | head -1)
[[ -n "$BEST_NAME" ]] && BEST="${CKPT_DIR}/${BEST_NAME}" || BEST=""
if [[ -n "$BEST" ]]; then
    cp "$BEST" checkpoints/uma-inverse-v5.ckpt
    echo "v5 final: $BEST -> checkpoints/uma-inverse-v5.ckpt"
else
    echo "WARNING: no top-K snapshot found in $CKPT_DIR; set --ckpt explicitly downstream"
fi

echo "v5 Stage 3 DDP2 complete."
