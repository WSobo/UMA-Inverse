#!/bin/bash
#SBATCH --job-name=uma-runA-distogram
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/runA_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/runA_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set — add it to ~/.bashrc}"

# ── RUN A — distogram-only probe, warm-started from v4 stage 3 ────────────────
#
# Purpose: isolate the encoder fix. This run is IDENTICAL to v4 stage 3 in every
# data/model flag — same v4 cache (data/processed/), ligand_context_atoms=25,
# all v3/v4 geometry featurization — and differs ONLY by:
#     ++model.distogram_aux_weight=0.2   (geometry aux supervision on Z_ij)
#     ++model.distogram_num_bins=38
# So any delta in recovery / cofold is attributable to the distogram head alone,
# NOT to the v5 chemistry / NA / M=50 changes (all still default-off here).
#
# Warm-start: weights-only from a frozen epoch-6 snapshot of the live v4 stage 3
# run (val_loss 1.2166). The distogram head has no saved weights and initialises
# fresh — train.py loads with strict=False but fails on any *other* missing/
# unexpected key, so a genuine architecture mismatch still errors out.
#
# Budget: single A100 (gradient checkpointing keeps N=384 under 80 GB at bsz=8),
# 3 epochs, restarted short schedule (warmup=500, T_max≈56k ≈ 3 epochs at bsz=8).
# Cheap by design — leaves the 2×A100 for the v4 stage 3 run still in flight.
#
# Track live: W&B val/distogram_top1 (should climb 0.266 → 0.5+ if the encoder
# can carry geometry) and val/ce_loss / recovery. Each epoch writes a checkpoint
# under checkpoints/pairmixerinv-runA-distogram-warmstart/ to benchmark.

INIT_CKPT=checkpoints/runA_v4_stage3_ep6_init.ckpt
echo ">> [RUN A] warm-starting distogram-only from: ${INIT_CKPT}"

uv run python scripts/train.py \
    run_name="pairmixerinv-runA-distogram-warmstart" \
    ++wandb.enabled=true \
    ++trainer.max_epochs=3 \
    ++data.max_total_nodes=384 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=true \
    ++training.devices=1 \
    ++training.warmup_steps=500 \
    ++training.T_max=56000 \
    ++trainer.init_from_checkpoint="${INIT_CKPT}" \
    ++data.pair_distance_atoms=backbone_full_25 \
    ++data.pair_distance_atoms_ligand=backbone_full \
    ++data.ligand_featurizer=ligandmpnn_atomic \
    ++data.frame_relative_angles=true \
    ++data.return_sidechain_atoms=true \
    ++data.sidechain_context_rate=0.03 \
    ++training.coord_noise_std=0.1 \
    ++data.ligand_context_atoms=25 \
    ++model.distogram_aux_weight=0.2 \
    ++model.distogram_num_bins=38

echo "Run A complete. Checkpoints under checkpoints/pairmixerinv-runA-distogram-warmstart/"
