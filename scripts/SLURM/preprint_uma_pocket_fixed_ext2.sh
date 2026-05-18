#!/bin/bash
#SBATCH --job-name=uma-pocket-ext2
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/uma_pocket_ext2_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/uma_pocket_ext2_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# UMA-Inverse v3 pocket-fixed redesign on the EXT2 25 small-mol PDBs
# (outputs/preprint/pdb_selection_extended2.json). Writes new <pdb_id>/
# subdirs into the existing uma_pocket_fixed_v3/ tree.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> UMA v3 pocket-fixed designs (ext2, 25 new PDBs)"
echo ">>   ckpt:      checkpoints/uma-inverse-v3.ckpt"
echo ">>   config:    configs/config_v3.yaml"
echo ">>   selection: outputs/preprint/pdb_selection_extended2.json"
echo ">>   out-dir:   outputs/preprint/uma_pocket_fixed_v3"

uv run python scripts/paper/run_pocket_fixed_designs.py \
    --ckpt checkpoints/uma-inverse-v3.ckpt \
    --config configs/config_v3.yaml \
    --selection outputs/preprint/pdb_selection_extended2.json \
    --out-dir outputs/preprint/uma_pocket_fixed_v3

echo "Done."
