#!/bin/bash
#SBATCH --job-name=uma-runA-control
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/runA_ctrl_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/runA_ctrl_%j.err
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

# ── RUN A — λ=0 CONTROL (matched twin of the distogram probe) ─────────────────
#
# Identical to 04a_runA_distogram_warmstart.sh in EVERY knob — same frozen v4
# stage-3 epoch-6 init, same single A100, bsz=8, max_total_nodes=384, grad
# checkpointing, 3 epochs, same restarted schedule (warmup=500, T_max=56k) —
# except the distogram aux head is OFF (++model.distogram_aux_weight=0.0).
#
# With weight 0.0 the head is never instantiated, so this run IS v4 (verified:
# flags-off == v4). It is therefore "v4 epoch-6 warm-started + 3 epochs" with
# the SAME batch size and LR schedule as Run A.
#
# Purpose: hold training time / batch / schedule constant so the recovery delta
#     Run A (λ=0.2)  −  this control (λ=0)   at the same epoch
# is attributable to the distogram head ALONE, not to "3 more epochs of
# training" or the smaller effective batch. Run them in parallel and benchmark
# matching epochs.
#
# Outputs are scoped by run_name (train.py derives checkpoints/{run_name}/,
# logs/csv/{run_name}/, W&B name={run_name}) so nothing collides with Run A.

INIT_CKPT=checkpoints/runA_v4_stage3_ep6_init.ckpt
echo ">> [RUN A — CONTROL λ=0] warm-starting from: ${INIT_CKPT}"

uv run python scripts/train.py \
    run_name="pairmixerinv-runA-control-warmstart" \
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
    ++model.distogram_aux_weight=0.0

echo "Run A control complete. Checkpoints under checkpoints/pairmixerinv-runA-control-warmstart/"
