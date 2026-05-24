#!/bin/bash
#SBATCH --job-name=cofold-metrics-ext2
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/cofold_metrics_ext2_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/cofold_metrics_ext2_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=medium

# Compute cofold metrics over the full 50-small-mol + 10-metal set (60 PDBs).
#
# Two-pass approach because cofold_metrics.py uses a single cofold_base dir:
#   Pass A: existing v3 results (35 UMA + 20 LMPNN from cofold_v3/)
#   Pass B: new ext2 results   (25 UMA + 40 LMPNN from cofold_v3_ext2/)
# Then merge into cofold_metrics_ext2.csv.
#
# Requires preprint_boltz_cofold_v3_ext2.sh to have finished.

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "${PROJ}"

SELECTION="${PROJ}/outputs/preprint/pdb_selection_combined2.json"
OUT_A="${PROJ}/outputs/preprint/cofold_metrics_ext2_pass_a.csv"
OUT_B="${PROJ}/outputs/preprint/cofold_metrics_ext2_pass_b.csv"
OUT_FINAL="${PROJ}/outputs/preprint/cofold_metrics_ext2.csv"

echo ">> cofold_metrics ext2 (pass A: existing v3 results)"
uv run python scripts/paper/cofold_metrics.py \
    --sampling-record "${PROJ}/outputs/preprint/boltz_inputs/cofold_v3/sampling_record_merged.json" \
    --cofold-base "${PROJ}/outputs/preprint/cofold_v3" \
    --selection "${SELECTION}" \
    --out "${OUT_A}"

echo ">> cofold_metrics ext2 (pass B: new ext2 results)"
uv run python scripts/paper/cofold_metrics.py \
    --sampling-record "${PROJ}/outputs/preprint/boltz_inputs/cofold_v3_ext2/sampling_record_merged.json" \
    --cofold-base "${PROJ}/outputs/preprint/cofold_v3_ext2" \
    --selection "${SELECTION}" \
    --out "${OUT_B}"

echo ">> Merging pass A + pass B into ${OUT_FINAL}"
uv run python -c "
import csv, pathlib
a = list(csv.DictReader(open('${OUT_A}')))
b = list(csv.DictReader(open('${OUT_B}')))
rows = a + b
if not rows:
    raise SystemExit('no rows -- something went wrong')
out = pathlib.Path('${OUT_FINAL}')
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'merged {len(a)} + {len(b)} = {len(rows)} rows -> {out}')
uma_pdbs = {r[\"pdb_id\"] for r in rows if r[\"method\"].startswith(\"uma\")}
lm_pdbs  = {r[\"pdb_id\"] for r in rows if r[\"method\"] == \"ligandmpnn\"}
print(f'UMA PDBs: {len(uma_pdbs)}   LigandMPNN PDBs: {len(lm_pdbs)}')
"

echo ">> cofold_metrics_ext2 complete."
