#!/bin/bash
#SBATCH --job-name=uma-inv-v5-stage2-1gpu
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/v5_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/v5_train_%j.err
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2

export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set — add it to ~/.bashrc}"

# v5 STAGE 2 — SINGLE generic GPU (gres=gpu:1), crash-immune like stage 1.
# Effective batch 16 held via bsz=8 × accumulate_grad_batches=2 on one GPU:
# same optimizer-steps/epoch and schedule (25 epochs, warmup=1k, T_max=230k) as
# the DDP2 design, at bsz=8 memory. gradient_checkpointing ON for A5500 24 GB.
# (run_name kept "...-ddp2" so Stage 3's checkpoint cross-reference resolves.)
# Distogram aux head warm-starts from Stage 1's last.ckpt.

STAGE1_DIR=checkpoints/pairmixerinv-v5-stage1-nodes64
STAGE1_BEST=$(ls "${STAGE1_DIR}"/uma-inverse-*-*.ckpt 2>/dev/null \
    | grep -Ev -- '-v[0-9]+\.ckpt$' \
    | awk -F'-' '{print $NF, $0}' | sort -n | head -1 | cut -d' ' -f2-)
STAGE1_CKPT="${STAGE1_BEST:-${STAGE1_DIR}/last.ckpt}"
echo ">> Stage 2 warm-starting from: ${STAGE1_CKPT}"

echo ">> [v5 STAGE 2 1GPU] 25 epochs at max_total_nodes=128, bsz=8 x accum2 (eff 16), devices=1"
uv run python scripts/train.py \
    run_name="pairmixerinv-v5-stage2-nodes128-ddp2" \
    ++wandb.enabled=true \
    ++trainer.max_epochs=25 \
    ++data.max_total_nodes=128 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=true \
    ++training.devices=1 \
    ++training.accumulate_grad_batches=2 \
    ++training.warmup_steps=1000 \
    ++training.T_max=230000 \
    ++trainer.init_from_checkpoint="${STAGE1_CKPT}" \
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

echo "v5 Stage 2 (1GPU) complete."
