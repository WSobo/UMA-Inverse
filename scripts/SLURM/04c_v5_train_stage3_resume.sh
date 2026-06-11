#!/bin/bash
#SBATCH --job-name=uma-inv-v5-stage3-resume
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

# v5 STAGE 3 DDP2 — RESUME variant. Intended to be sbatch'd with
#   --dependency=afterany:<stage3_jobid>
# so it fires the moment the primary Stage 3 job ends (wall-clock TIMEOUT,
# crash, or clean stop). Uses resume_from_checkpoint (FULL state: weights +
# optimizer + LR-scheduler step + global_step), NOT init_from_checkpoint —
# so the cosine schedule continues from where it left off (~step 110k) instead
# of resetting to step 0. Same run_name/ckpt dir as the primary so top-K
# tracking and the promote-best tail keep working across the boundary.
#
# last.ckpt is resolved at job start (after the dependency clears), so it picks
# up the freshest epoch the primary job wrote before stopping.

RUN_DIR=checkpoints/pairmixerinv-v5-stage3-nodes384-ddp2
RESUME_CKPT="${RUN_DIR}/last.ckpt"
if [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "FATAL: resume checkpoint not found: ${RESUME_CKPT}" >&2
    exit 1
fi

# Verify last.ckpt is the late-stage epoch we expect (the primary should have
# reached ~18-19 before the wall). Reading ck['epoch'] is the only reliable way
# to confirm — last.ckpt has no epoch in its filename. Abort if it looks stale
# or early (< 12), which would mean we're about to resume from the wrong place.
RESUME_EPOCH=$(uv run python -c "import torch; print(torch.load('${RESUME_CKPT}', map_location='cpu', weights_only=False)['epoch'])")
echo ">> Stage 3 RESUME from full state: ${RESUME_CKPT} (epoch=${RESUME_EPOCH})"
if ! [[ "${RESUME_EPOCH}" =~ ^[0-9]+$ ]] || [[ "${RESUME_EPOCH}" -lt 12 ]]; then
    echo "FATAL: resume epoch '${RESUME_EPOCH}' < 12 — checkpoint looks stale/wrong; aborting." >&2
    exit 1
fi

echo ">> [v5 STAGE 3 DDP2 RESUME] -> 24 epochs at max_total_nodes=384, bsz=8/rank, devices=2 (A100)"
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
    ++trainer.resume_from_checkpoint="${RESUME_CKPT}" \
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

echo "v5 Stage 3 DDP2 RESUME complete."
