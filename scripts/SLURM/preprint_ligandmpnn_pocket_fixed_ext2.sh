#!/bin/bash
#SBATCH --job-name=ligmpnn-pocket-ext2
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/ligmpnn_pocket_ext2_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/ligmpnn_pocket_ext2_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# LigandMPNN pocket-fixed redesign on the EXT2 25 small-mol PDBs.
# Writes new <pdb>.fa files into ligandmpnn_pocket_fixed/seqs/ alongside
# the existing 35. Same protocol as prior ext runs: K=20, T=0.1, seed=0.

set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate ligandmpnn_env

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
LIGANDMPNN_DIR=/private/groups/yehlab/wsobolew/01_software/LigandMPNN
INPUTS_DIR="${PROJ}/outputs/preprint/ligandmpnn_inputs_ext2"
OUT_DIR="${PROJ}/outputs/preprint/ligandmpnn_pocket_fixed"

mkdir -p "${OUT_DIR}"

CHECKPOINT_PATH="${LIGANDMPNN_DIR}/model_params/ligandmpnn_v_32_010_25.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "!! LigandMPNN weights not found at ${CHECKPOINT_PATH}" >&2
    exit 1
fi

echo ">> Building LigandMPNN inputs for ext2 (25 PDBs)"
cd "${PROJ}"
uv run python scripts/paper/build_ligandmpnn_inputs.py \
    --selection outputs/preprint/pdb_selection_extended2.json \
    --out-dir "${INPUTS_DIR}"

cd "${LIGANDMPNN_DIR}"

echo ">> LigandMPNN pocket-fixed designs (ext2)"
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
