#!/bin/bash
#SBATCH --job-name=uma-inv-v4-stage2-ddp2
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/v4_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/v4_train_%j.err
#SBATCH --time=48:00:00
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

# v4 STAGE 2 DDP2 — 2×A100, max_total_nodes=128. bsz=8/rank gives effective
# batch=16, matching v3 Stage 2 (4×A5500 bsz=4/rank=16). A100 VRAM (~40 GB)
# comfortably handles bsz=8 at nodes=128 without gradient checkpointing.
# Same schedule as v3 Stage 2 (25 epochs, warmup=1k, T_max=230k).

STAGE1_DIR=checkpoints/pairmixerinv-v4-stage1-nodes64
STAGE1_BEST=$(ls "${STAGE1_DIR}"/uma-inverse-*-*.ckpt 2>/dev/null \
    | grep -Ev -- '-v[0-9]+\.ckpt$' \
    | awk -F'-' '{print $NF, $0}' | sort -n | head -1 | cut -d' ' -f2-)
STAGE1_CKPT="${STAGE1_BEST:-${STAGE1_DIR}/last.ckpt}"
echo ">> Stage 2 warm-starting from: ${STAGE1_CKPT}"

echo ">> [v4 STAGE 2 DDP2] 25 epochs at max_total_nodes=128, bsz=8/rank, devices=2 (A100)"
srun uv run python scripts/train.py \
    run_name="pairmixerinv-v4-stage2-nodes128-ddp2" \
    ++wandb.enabled=true \
    ++trainer.max_epochs=25 \
    ++data.max_total_nodes=128 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.devices=2 \
    ++training.warmup_steps=1000 \
    ++training.T_max=230000 \
    ++trainer.init_from_checkpoint="${STAGE1_CKPT}" \
    ++data.pair_distance_atoms=backbone_full_25 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.ligand_featurizer=ligandmpnn_atomic \
    ++data.frame_relative_angles=true \
    ++data.return_sidechain_atoms=true \
    ++data.sidechain_context_rate=0.03 \
    ++training.coord_noise_std=0.1

echo "v4 Stage 2 DDP2 complete."
