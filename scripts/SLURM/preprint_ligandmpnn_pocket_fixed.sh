#!/bin/bash
#SBATCH --job-name=ligmpnn-pocket-fixed
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/ligmpnn_pocket_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/ligmpnn_pocket_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# LigandMPNN pocket-fixed redesign over the 20 PDBs from
# outputs/preprint/pdb_selection.json. K=20 sequences per PDB
# (number_of_batches=20, batch_size=1), T=0.1, random decoding (default).
#
# Uses the user's existing LigandMPNN install (NOT the vendored project copy),
# matching scripts/ligandMPNN_example.ipynb conventions:
#     - Install: /private/groups/yehlab/wsobolew/01_software/LigandMPNN
#     - Env:     micromamba activate ligandmpnn_env
#     - Weights: ligandmpnn_v_32_010_25.pt (the canonical paper checkpoint)

set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate ligandmpnn_env

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
LIGANDMPNN_DIR=/private/groups/yehlab/wsobolew/01_software/LigandMPNN
INPUTS_DIR="${PROJ}/outputs/preprint/ligandmpnn_inputs"
OUT_DIR="${PROJ}/outputs/preprint/ligandmpnn_pocket_fixed"

# Build inputs JSON if not present (idempotent)
if [[ ! -f "${INPUTS_DIR}/pdb_path_multi.json" || ! -f "${INPUTS_DIR}/fixed_residues_multi.json" ]]; then
    echo ">> Building LigandMPNN inputs..."
    cd "${PROJ}"
    uv run python scripts/preprint/build_ligandmpnn_inputs.py
fi

mkdir -p "${OUT_DIR}"

CHECKPOINT_PATH="${LIGANDMPNN_DIR}/model_params/ligandmpnn_v_32_010_25.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "!! LigandMPNN weights not found at ${CHECKPOINT_PATH}" >&2
    exit 1
fi

cd "${LIGANDMPNN_DIR}"

echo ">> LigandMPNN pocket-fixed designs"
echo ">>   inputs:  ${INPUTS_DIR}"
echo ">>   output:  ${OUT_DIR}"
echo ">>   weights: ${CHECKPOINT_PATH}"

python run.py \
    --model_type ligand_mpnn \
    --checkpoint_ligand_mpnn "${CHECKPOINT_PATH}" \
    --pdb_path_multi "${INPUTS_DIR}/pdb_path_multi.json" \
    --fixed_residues_multi "${INPUTS_DIR}/fixed_residues_multi.json" \
    --chains_to_design A \
    --out_folder "${OUT_DIR}" \
    --batch_size 1 \
    --number_of_batches 20 \
    --temperature 0.1 \
    --seed 0 \
    --save_stats 1

echo "Done."
