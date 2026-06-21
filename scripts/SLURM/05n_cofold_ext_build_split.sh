#!/bin/bash
#SBATCH --job-name=cofold-ext-buildsplit
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/cofold_ext_buildsplit_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/cofold_ext_buildsplit_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short

# Stage 2: build the full 3-arm Boltz YAMLs over the 104-PDB combined set
# (deterministic seed=42 -> the existing 35 regenerate identically, 69 new added),
# then split the NOT-yet-cofolded records into K chunks for the parallel array.
# CPU-only (no Boltz here).

set -e
PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "$PROJ"
COMB=outputs/preprint/pdb_selection_combined_v5.json
K="${K:-8}"

echo ">> [1/2] build_cofold_yamls (combined_v5, 104 PDBs)  ($(date))"
uv run python scripts/paper/build_cofold_yamls.py \
    --selection "$COMB" \
    --uma-dir outputs/preprint/uma_pocket_fixed_v5 \
    --uma-method-name uma_v5 \
    --ligandmpnn-dir outputs/preprint/ligandmpnn_pocket_fixed \
    --include-native \
    --samples-per-pdb 3

echo ">> [2/2] splitting not-yet-cofolded records into $K chunks  ($(date))"
uv run python - "$K" <<'PY'
import sys, os, shutil
from pathlib import Path
K = int(sys.argv[1])
inb = Path("outputs/preprint/boltz_inputs/cofold")
cof = Path("outputs/preprint/cofold")
chunks = Path("outputs/preprint/boltz_inputs/cofold_chunks")
if chunks.exists():
    shutil.rmtree(chunks)
methods = ["uma_v5", "native", "ligandmpnn"]
todo = []
for m in methods:
    for y in sorted((inb / m).glob("*.yaml")):
        pred = cof / m / f"boltz_results_{m}" / "predictions" / y.stem
        if not pred.exists():
            todo.append((m, y))
print(f"records already cofolded are skipped; NEW records to cofold: {len(todo)}")
for i, (m, y) in enumerate(todo):
    d = chunks / f"chunk_{i % K}" / m
    d.mkdir(parents=True, exist_ok=True)
    os.symlink(y.resolve(), d / y.name)
for k in range(K):
    n = sum(len(list((chunks / f"chunk_{k}" / m).glob("*.yaml")))
            for m in methods if (chunks / f"chunk_{k}" / m).exists())
    print(f"  chunk_{k}: {n} yamls")
PY
echo ">> Stage 2 (build+split) complete  ($(date))"
