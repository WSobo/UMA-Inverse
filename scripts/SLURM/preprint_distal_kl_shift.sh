#!/bin/bash
#SBATCH --job-name=distal-kl
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/distal_kl_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/distal_kl_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Layer 3 — distal-residue ligand-conditioning KL shift, UMA-v3 vs LigandMPNN.
#
# Two invocation modes selected by KL_MODE:
#   mechanism (default): ~200 PDBs sampled from the LigandMPNN val split,
#                        restricted to <=359 residues to avoid v3 cropping.
#                        Used for the main mechanism figure.
#   outcome:             the curated 35-PDB combined set
#                        (pdb_selection_combined.json). Cross-experiment
#                        consistent with pocket-fixed redesign.
#
# Override via:
#   sbatch --export=ALL,KL_MODE=outcome scripts/SLURM/preprint_distal_kl_shift.sh
#   sbatch --export=ALL,KL_CKPT=path/v3.ckpt,KL_N=400 scripts/SLURM/preprint_distal_kl_shift.sh

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2
cd "${PROJ}"

# Defaults — override via sbatch --export
KL_CKPT="${KL_CKPT:-checkpoints/uma-inverse-v3.ckpt}"
KL_CONFIG="${KL_CONFIG:-configs/config_v3.yaml}"   # must match the ckpt's training featurization
KL_MODE="${KL_MODE:-mechanism}"
KL_N="${KL_N:-200}"
KL_SEED="${KL_SEED:-0}"
KL_NUM_BATCHES="${KL_NUM_BATCHES:-10}"
KL_EXTRA_ARGS="${KL_EXTRA_ARGS:-}"                 # e.g. "--skip-ligandmpnn"

# PDB list selection
case "${KL_MODE}" in
    mechanism)
        # Sample KL_N PDB IDs from the LigandMPNN val split.
        VAL_JSON="${PROJ}/LigandMPNN/training/valid.json"
        OUT_DIR_DEFAULT="${PROJ}/outputs/preprint/distal_kl/mechanism"
        OUT_DIR="${KL_OUT_DIR:-${OUT_DIR_DEFAULT}}"
        mkdir -p "${OUT_DIR}"
        PDB_LIST="${OUT_DIR}/pdb_list.txt"
        if [[ ! -f "${PDB_LIST}" ]]; then
            echo ">> Sampling ${KL_N} PDB IDs from ${VAL_JSON}"
            uv run python -c "
import json, random, pathlib
val = json.loads(pathlib.Path('${VAL_JSON}').read_text())
ids = list(val.keys()) if isinstance(val, dict) else list(val)
random.Random(${KL_SEED}).shuffle(ids)
out = pathlib.Path('${PDB_LIST}')
out.write_text('\n'.join(ids[:${KL_N}]) + '\n')
print(f'wrote {out} ({len(ids[:${KL_N}])} ids)')
"
        fi
        ;;
    outcome)
        OUT_DIR_DEFAULT="${PROJ}/outputs/preprint/distal_kl/outcome"
        OUT_DIR="${KL_OUT_DIR:-${OUT_DIR_DEFAULT}}"
        mkdir -p "${OUT_DIR}"
        PDB_LIST="${OUT_DIR}/pdb_list.txt"
        if [[ ! -f "${PDB_LIST}" ]]; then
            echo ">> Building PDB list from pdb_selection_combined.json"
            uv run python -c "
import json, pathlib
sel = json.loads(pathlib.Path('${PROJ}/outputs/preprint/pdb_selection_combined.json').read_text())
ids = [e['pdb_id'] for e in sel['small_molecule']] + [e['pdb_id'] for e in sel['metal']]
out = pathlib.Path('${PDB_LIST}')
out.write_text('\n'.join(ids) + '\n')
print(f'wrote {out} ({len(ids)} ids)')
"
        fi
        ;;
    *)
        echo "Unknown KL_MODE: ${KL_MODE}. Expected 'mechanism' or 'outcome'." >&2
        exit 1
        ;;
esac

echo ">> distal-KL config:"
echo "     ckpt        = ${KL_CKPT}"
echo "     mode        = ${KL_MODE}"
echo "     pdb list    = ${PDB_LIST}"
echo "     out dir     = ${OUT_DIR}"
echo "     num_batches = ${KL_NUM_BATCHES}"
echo "     seed        = ${KL_SEED}"

uv run python scripts/paper/distal_kl_shift.py \
    --uma-ckpt "${KL_CKPT}" \
    --config "${KL_CONFIG}" \
    --pdb-list "${PDB_LIST}" \
    --out-dir "${OUT_DIR}" \
    --num-batches "${KL_NUM_BATCHES}" \
    --seed "${KL_SEED}" \
    ${KL_EXTRA_ARGS}

echo ">> distal-KL run complete."
