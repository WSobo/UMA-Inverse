#!/bin/bash
#SBATCH --job-name=uma-pocket-ext
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/uma_pocket_ext_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/uma_pocket_ext_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# UMA-Inverse pocket-fixed redesign on the EXTENDED 15 small-mol PDBs
# (outputs/preprint/pdb_selection_extended.json). Output goes into the same
# uma_pocket_fixed/ tree as the original 10 -- new <pdb_id>/ subdirs only.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> UMA pocket-fixed designs (extended)"
echo ">>   ckpt: checkpoints/pairmixerinv-v2-stage3-nodes384-ddp8/uma-inverse-19-1.1463.ckpt"
echo ">>   selection: outputs/preprint/pdb_selection_extended.json"

uv run python scripts/paper/run_pocket_fixed_designs.py \
    --selection outputs/preprint/pdb_selection_extended.json \
    "$@"

echo "Done."
