#!/bin/bash
#SBATCH --job-name=uma-inferinv
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/infer_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/infer_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo "Running Fast UMA-Inverse Inference..."
# Update the paths below according to your exact needs!
uv run python scripts/inference.py \
    --pdb ../LigandMPNN/inputs/1BC8.pdb \
    --config configs/config.yaml \
    --ckpt checkpoints/last.ckpt
