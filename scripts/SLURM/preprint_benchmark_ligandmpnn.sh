#!/bin/bash
#SBATCH --job-name=lmpnn-benchmark
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/lmpnn_benchmark_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/lmpnn_benchmark_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Evaluate LigandMPNN on the same val.json split used by the UMA-Inverse
# benchmark, for an apples-to-apples data-split comparison.
#
# Runs LigandMPNN at T=0.1 with --save_stats 1 (one sample per PDB), then
# post-processes stats/*.pt files into per_pdb.csv + summary.md/json.
#
# NOTE: LigandMPNN decodes autoregressively at T=0.1; UMA-Inverse reports
# teacher-forced recovery (argmax given full native context). These differ
# systematically -- teacher-forced is higher. See postprocess script header.
#
# Outputs: outputs/benchmark/ligandmpnn-val2000/

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
LIGANDMPNN_DIR=/private/groups/yehlab/wsobolew/01_software/LigandMPNN

VAL_JSON="${PROJ}/LigandMPNN/training/valid.json"
PDB_DIR="${PROJ}/data/raw/pdb_archive"
N_PDBS="${BENCH_N:-2000}"
OUT_DIR="${PROJ}/outputs/benchmark/ligandmpnn-val${N_PDBS}"
INPUTS_DIR="${OUT_DIR}/inputs"
STATS_DIR="${OUT_DIR}/stats"

mkdir -p "${INPUTS_DIR}" "${STATS_DIR}"

export PROJ N_PDBS

CHECKPOINT_PATH="${LIGANDMPNN_DIR}/model_params/ligandmpnn_v_32_010_25.pt"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "!! LigandMPNN weights not found at ${CHECKPOINT_PATH}" >&2
    exit 1
fi

# ── Activate ligandmpnn_env (needed for ProDy pre-validation and LigandMPNN) ──
eval "$(micromamba shell hook --shell bash)"
micromamba activate ligandmpnn_env

# ── Resolve + pre-validate PDB paths using ProDy (same env as LigandMPNN) ────
# Running validation under ligandmpnn_env catches every ProDy failure mode:
# DNA/RNA-only structures, shape mismatches, non-standard files, etc.
echo ">> Resolving and ProDy-validating ${N_PDBS} PDB paths from val.json"
cd "${PROJ}"
python - <<PYEOF
import json, sys
from pathlib import Path

proj    = Path("${PROJ}")
val_ids = json.loads((proj / "LigandMPNN/training/valid.json").read_text())
pdb_dir = proj / "data/raw/pdb_archive"
n_pdbs  = int("${N_PDBS}")

def resolve(pdb_id):
    pid = pdb_id.lower()
    for p in [pdb_dir / pid[1:3] / f"{pid}.pdb", pdb_dir / f"{pid}.pdb"]:
        if p.exists():
            return p
    return None

from prody import parsePDB, confProDy
confProDy(verbosity="none")

def prody_ok(path):
    try:
        atoms = parsePDB(str(path))
        return atoms is not None and atoms.select("protein and backbone") is not None
    except Exception:
        return False

pdb_map = {}
n_missing = 0
n_invalid = 0
for pdb_id in val_ids:
    if len(pdb_map) >= n_pdbs:
        break
    path = resolve(pdb_id)
    if path is None:
        n_missing += 1
        continue
    if prody_ok(path):
        pdb_map[str(path)] = ""
    else:
        n_invalid += 1
        print(f"  skip (ProDy): {pdb_id}", file=sys.stderr)

# Write one chunk JSON per 100 PDBs so a single bad PDB only kills its chunk
paths = list(pdb_map.keys())
chunk_size = 100
chunks_dir = proj / "outputs/benchmark" / f"ligandmpnn-val{n_pdbs}" / "inputs" / "chunks"
chunks_dir.mkdir(parents=True, exist_ok=True)
chunk_files = []
for i in range(0, len(paths), chunk_size):
    chunk = {p: "" for p in paths[i:i+chunk_size]}
    cf = chunks_dir / f"chunk_{i//chunk_size:03d}.json"
    cf.write_text(json.dumps(chunk, indent=2))
    chunk_files.append(str(cf))

# Write the full map too (for reference)
out = proj / "outputs/benchmark" / f"ligandmpnn-val{n_pdbs}" / "inputs" / "pdb_path_multi.json"
out.write_text(json.dumps(pdb_map, indent=2))
# Write chunk list for bash loop
(chunks_dir / "chunk_list.txt").write_text("\n".join(chunk_files) + "\n")
print(f"resolved {len(pdb_map)} PDBs (missing={n_missing}, prody_invalid={n_invalid})")
print(f"split into {len(chunk_files)} chunks of {chunk_size} -> {chunks_dir}")
PYEOF

N_RESOLVED=$(python3 -c "import json; d=json.load(open('${INPUTS_DIR}/pdb_path_multi.json')); print(len(d))")
N_CHUNKS=$(wc -l < "${INPUTS_DIR}/chunks/chunk_list.txt")
echo ">> Running LigandMPNN on ${N_RESOLVED} PDBs in ${N_CHUNKS} chunks (T=0.1, 1 sample, save_stats)"

cd "${LIGANDMPNN_DIR}"

n_ok=0
n_fail=0
while IFS= read -r chunk_json; do
    chunk_name=$(basename "${chunk_json}" .json)
    echo "   chunk ${chunk_name} ..."
    if python run.py \
        --model_type ligand_mpnn \
        --checkpoint_ligand_mpnn "${CHECKPOINT_PATH}" \
        --pdb_path_multi "${chunk_json}" \
        --out_folder "${OUT_DIR}" \
        --batch_size 1 \
        --number_of_batches 1 \
        --temperature 0.1 \
        --seed 0 \
        --save_stats 1 2>/dev/null; then
        n_ok=$((n_ok + 1))
    else
        echo "   !! chunk ${chunk_name} failed -- skipping"
        n_fail=$((n_fail + 1))
    fi
done < "${INPUTS_DIR}/chunks/chunk_list.txt"

echo ">> LigandMPNN done: ${n_ok} chunks OK, ${n_fail} chunks failed"

echo ">> LigandMPNN inference done. Post-processing stats..."

cd "${PROJ}"
uv run python scripts/paper/benchmark_ligandmpnn_postprocess.py \
    --stats-dir "${OUT_DIR}/stats" \
    --pdb-dir   "${PDB_DIR}" \
    --out-dir   "${OUT_DIR}"

echo ">> Done. Results in ${OUT_DIR}/summary.md"
cat "${OUT_DIR}/summary.md"
