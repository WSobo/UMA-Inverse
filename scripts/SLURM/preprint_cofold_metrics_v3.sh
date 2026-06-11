#!/bin/bash
#SBATCH --job-name=cofold-metrics-v3
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/cofold_metrics_v3_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/cofold_metrics_v3_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=medium

# Parse Boltz-2 v3 cofold outputs and write outputs/preprint/cofold_metrics_v3.csv.
# CPU-only — just JSON + PDB parsing + Kabsch alignment.

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2
cd "${PROJ}"

SAMPLING="${SAMPLING:-${PROJ}/outputs/preprint/boltz_inputs/cofold_v3/sampling_record.json}"
COFOLD_BASE="${COFOLD_BASE:-${PROJ}/outputs/preprint/cofold_v3}"
SELECTION="${SELECTION:-${PROJ}/outputs/preprint/pdb_selection_combined.json}"
OUT_CSV="${OUT_CSV:-${PROJ}/outputs/preprint/cofold_metrics_v3.csv}"

echo ">> cofold_metrics v3"
echo "   sampling: ${SAMPLING}"
echo "   cofold:   ${COFOLD_BASE}"
echo "   selection:${SELECTION}"
echo "   out:      ${OUT_CSV}"

# Append the LigandMPNN sampling records from the v2 run so the resulting
# CSV has both methods present (uma_v3 + ligandmpnn paired per PDB).
LMPNN_SAMPLING="${PROJ}/outputs/preprint/boltz_inputs/cofold/sampling_record.json"
MERGED_SAMPLING="${PROJ}/outputs/preprint/boltz_inputs/cofold_v3/sampling_record_merged.json"
if [[ -f "${LMPNN_SAMPLING}" && -f "${SAMPLING}" ]]; then
    uv run python -c "
import json, pathlib
v3 = json.loads(pathlib.Path('${SAMPLING}').read_text())
lm = json.loads(pathlib.Path('${LMPNN_SAMPLING}').read_text())
lm = [r for r in lm if r.get('method') == 'ligandmpnn']
out = pathlib.Path('${MERGED_SAMPLING}')
out.write_text(json.dumps(v3 + lm, indent=2))
print(f'merged sampling record: {len(v3)} v3 + {len(lm)} ligandmpnn -> {out}')
"
    SAMPLING="${MERGED_SAMPLING}"
fi

uv run python scripts/paper/cofold_metrics.py \
    --sampling-record "${SAMPLING}" \
    --cofold-base "${COFOLD_BASE}" \
    --selection "${SELECTION}" \
    --out "${OUT_CSV}"

echo ">> cofold_metrics_v3 complete."
