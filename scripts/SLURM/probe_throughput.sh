#!/bin/bash
#SBATCH --job-name=umainv-probe
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/probe_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/probe_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

# Short throughput probe at max_total_nodes=128, batch_size=8, grad_ckpt=off.
# Goal: confirm VRAM headroom and measure it/s before bumping stage 2 in
# 03_train_model.sh.

echo ">> Probe: N=128 bsz=8 grad_ckpt=off"
uv run python -u scripts/train.py \
    run_name="throughput-probe-n128-bsz8" \
    ++data.max_total_nodes=128 \
    ++data.batch_size=8 \
    ++data.num_workers=8 \
    ++model.gradient_checkpointing=false \
    ++training.warmup_steps=100 \
    ++training.T_max=5000 \
    ++trainer.max_epochs=1

echo "Probe complete. Check logs/csv/throughput-probe-n128-bsz8/ and nvidia-smi in the out log."
