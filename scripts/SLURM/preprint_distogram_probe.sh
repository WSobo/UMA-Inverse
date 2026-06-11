#!/bin/bash
#SBATCH --job-name=distogram-probe
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/distogram_probe_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/distogram_probe_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# v3 retro diagnostic — Cβ-Cβ distogram linear probe on the frozen v3 trunk.
#
# Outcome thresholds (committed pre-read):
#   top-1 > 0.85   → encoder geometry strong  (v4: decoder/data)
#   top-1 ∈ [.6,.85] → encoder geometry partial (v4: feature density)
#   top-1 < 0.60   → encoder geometry weak    (v4: trunk rethink)
#
# Override via:
#   sbatch --export=ALL,PROBE_CKPT=path/v3.ckpt,PROBE_N=400 scripts/SLURM/preprint_distogram_probe.sh

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2
cd "${PROJ}"

PROBE_CKPT="${PROBE_CKPT:-checkpoints/uma-inverse-v3.ckpt}"
PROBE_CONFIG="${PROBE_CONFIG:-configs/config_v3.yaml}"
PROBE_N="${PROBE_N:-400}"
PROBE_SEED="${PROBE_SEED:-0}"
PROBE_OUT_DIR="${PROBE_OUT_DIR:-${PROJ}/outputs/preprint/distogram_probe}"
PROBE_EPOCHS="${PROBE_EPOCHS:-10}"
PROBE_PAIRS_PER_PDB="${PROBE_PAIRS_PER_PDB:-4000}"

mkdir -p "${PROBE_OUT_DIR}"

PDB_LIST="${PROBE_OUT_DIR}/pdb_list.txt"
VAL_JSON="${PROJ}/LigandMPNN/training/valid.json"

if [[ ! -f "${PDB_LIST}" ]]; then
    echo ">> Sampling ${PROBE_N} PDB IDs from ${VAL_JSON}"
    uv run python -c "
import json, random, pathlib
val = json.loads(pathlib.Path('${VAL_JSON}').read_text())
ids = list(val.keys()) if isinstance(val, dict) else list(val)
random.Random(${PROBE_SEED}).shuffle(ids)
out = pathlib.Path('${PDB_LIST}')
out.write_text('\n'.join(ids[:${PROBE_N}]) + '\n')
print(f'wrote {out} ({len(ids[:${PROBE_N}])} ids)')
"
fi

echo ">> distogram-probe config:"
echo "     ckpt        = ${PROBE_CKPT}"
echo "     config      = ${PROBE_CONFIG}"
echo "     pdb list    = ${PDB_LIST}"
echo "     out dir     = ${PROBE_OUT_DIR}"
echo "     epochs      = ${PROBE_EPOCHS}"
echo "     pairs/PDB   = ${PROBE_PAIRS_PER_PDB}"
echo "     seed        = ${PROBE_SEED}"

uv run python scripts/paper/distogram_probe.py \
    --uma-ckpt "${PROBE_CKPT}" \
    --config "${PROBE_CONFIG}" \
    --pdb-list "${PDB_LIST}" \
    --out-dir "${PROBE_OUT_DIR}" \
    --epochs "${PROBE_EPOCHS}" \
    --pairs-per-pdb "${PROBE_PAIRS_PER_PDB}" \
    --seed "${PROBE_SEED}"

echo ">> distogram-probe run complete."
echo ">> Verdict:"
uv run python -c "
import json, pathlib
m = json.loads(pathlib.Path('${PROBE_OUT_DIR}/distogram_probe_metrics.json').read_text())
print(f\"  top1          = {m['top1']:.3f}\")
print(f\"  top3          = {m['top3']:.3f}\")
print(f\"  neighbor      = {m['neighbor_top1']:.3f}\")
print(f\"  expected MAE  = {m['expected_dist_mae_A']:.2f} Å\")
print(f\"  ECE           = {m['ece']:.3f}\")
print(f\"  VERDICT       = {m['verdict']}\")
"
