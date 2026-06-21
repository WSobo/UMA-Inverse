#!/bin/bash
#SBATCH --job-name=uma-inv-v5-stage2-resume4
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/v5_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/v5_train_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set — add it to ~/.bashrc}"

# v5 STAGE 2 RESUME — 4×A5500 (same node), TRUE mid-stage resume of the
# single-GPU run that was crawling at ~3.8 h/epoch. Picks up at global_step
# 55218 / epoch 5 with optimizer momentum + LR-schedule position restored.
#
# Effective batch held at 16 (bsz=4 × 4 ranks × accum=1) — IDENTICAL to the
# single-GPU bsz=8×accum2 design, so steps/epoch (~9203), T_max=230000, and the
# whole cosine curve are unchanged; only wall-clock parallelises (~3.3× faster).
# run_name kept identical so checkpoints continue in the SAME dir and Stage 3's
# STAGE2_DIR cross-reference resolves.
#
# Crash safety: num_workers=4/rank (16 total, vs the 32 that stalled the 4×A100
# run), DDP collective timeout raised to 2h in train.py, epoch checkpoints +
# resume_from_checkpoint make any stall cost ≤1 epoch.

RESUME_CKPT=checkpoints/pairmixerinv-v5-stage2-nodes128-ddp2/last.ckpt
echo ">> [v5 STAGE 2 RESUME 4×A5500] resuming from: ${RESUME_CKPT}"

srun uv run python scripts/train.py \
    run_name="pairmixerinv-v5-stage2-nodes128-ddp2" \
    ++wandb.enabled=true \
    ++trainer.max_epochs=25 \
    ++data.max_total_nodes=128 \
    ++data.batch_size=4 \
    ++data.num_workers=4 \
    ++model.gradient_checkpointing=true \
    ++training.devices=4 \
    ++training.accumulate_grad_batches=1 \
    ++training.warmup_steps=1000 \
    ++training.T_max=230000 \
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

echo "v5 Stage 2 (resume 4×A5500) complete."
