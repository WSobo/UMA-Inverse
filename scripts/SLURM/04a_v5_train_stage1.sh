#!/bin/bash
#SBATCH --job-name=uma-inv-v5-stage1-1gpu
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/v5_train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/v5_train_%j.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set — add it to ~/.bashrc}"

# v5 STAGE 1 — SINGLE generic GPU (gres=gpu:1 takes A5500 OR A100, whichever
# frees first — single slots open in ~hours vs ~3 days for 2-on-a-node). This
# config is CRASH-IMMUNE: the NCCL all-reduce that killed the 4-GPU run only
# exists with >1 GPU. effective batch 8 (bsz=8, devices=1) — unchanged from the
# original design, so schedule (15 epochs, warmup=1k) is identical.
# gradient_checkpointing ON so it fits A5500's 24 GB (the A100 run used ~20 GB
# un-checkpointed); checkpointing is math-neutral — same gradients, just slower.
#
# v5 deltas vs v4: chemistry OFF; ligand_context_atoms=50; distogram aux (0.2).

echo ">> [v5 STAGE 1 1GPU] 15 epochs at max_total_nodes=64, bsz=8, devices=1 (generic GPU)"
uv run python scripts/train.py \
    run_name="pairmixerinv-v5-stage1-nodes64" \
    ++wandb.enabled=true \
    ++trainer.max_epochs=15 \
    ++data.max_total_nodes=64 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=true \
    ++training.devices=1 \
    ++training.warmup_steps=1000 \
    ++training.T_max=280000 \
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

echo "v5 Stage 1 (1GPU) complete. last.ckpt at checkpoints/pairmixerinv-v5-stage1-nodes64/"
