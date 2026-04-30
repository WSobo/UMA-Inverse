#!/bin/bash
#SBATCH --job-name=ligmpnn-pocket-fixed
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/ligmpnn_pocket_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/ligmpnn_pocket_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# LigandMPNN pocket-fixed redesign over the 20 PDBs from
# outputs/preprint/pdb_selection.json. K=20 sequences per PDB
# (batch_size 4 x number_of_batches 5), T=0.1, random decoding order
# (the default for LigandMPNN).
#
# Prerequisite: LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt must be
# downloaded. This is the official LigandMPNN paper checkpoint (num_edges=32,
# atom_context_num=25, trained at 0.1A noise). Run get_model_params.sh once
# in the LigandMPNN dir (or just download that single weight file via wget)
# before launching this job.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

INPUTS_DIR="outputs/preprint/ligandmpnn_inputs"
OUT_DIR="outputs/preprint/ligandmpnn_pocket_fixed"

if [[ ! -f LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt ]]; then
    echo "!! LigandMPNN weights not found. Run:" >&2
    echo "    bash LigandMPNN/get_model_params.sh ./LigandMPNN/model_params" >&2
    exit 1
fi

if [[ ! -f "${INPUTS_DIR}/pdb_path_multi.json" || ! -f "${INPUTS_DIR}/fixed_residues_multi.json" ]]; then
    echo ">> Building LigandMPNN inputs..."
    uv run python scripts/preprint/build_ligandmpnn_inputs.py
fi

mkdir -p "${OUT_DIR}"

echo ">> LigandMPNN pocket-fixed designs"
echo ">>   inputs: ${INPUTS_DIR}"
echo ">>   output: ${OUT_DIR}"

# The UMA-Inverse env has all required deps for LigandMPNN/run.py (torch + numpy).
# Run from project root so relative ./model_params/ works.
cd LigandMPNN
uv run python run.py \
    --model_type ligand_mpnn \
    --pdb_path_multi "../${INPUTS_DIR}/pdb_path_multi.json" \
    --fixed_residues_multi "../${INPUTS_DIR}/fixed_residues_multi.json" \
    --out_folder "../${OUT_DIR}" \
    --batch_size 4 \
    --number_of_batches 5 \
    --temperature 0.1 \
    --seed 0 \
    --save_stats 1

echo "Done."
