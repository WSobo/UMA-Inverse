#!/bin/bash
#SBATCH --job-name=uma-pocket-fixed
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/uma_pocket_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/uma_pocket_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# UMA-Inverse pocket-fixed redesign over the 20 PDBs from
# outputs/preprint/pdb_selection.json. ~3-4h on 1x A5500.
# K=20 sequences per PDB, T=0.1, random decoding order.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> UMA pocket-fixed designs"
echo ">>   ckpt: checkpoints/pairmixerinv-v2-stage3-nodes384-ddp8/uma-inverse-19-1.1463.ckpt"
echo ">>   selection: outputs/preprint/pdb_selection.json"

uv run python scripts/preprint/run_pocket_fixed_designs.py "$@"

echo "Done."
