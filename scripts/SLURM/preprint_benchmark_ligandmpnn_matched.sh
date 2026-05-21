#!/bin/bash
#SBATCH --job-name=lmpnn-matched
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/lmpnn_matched_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/lmpnn_matched_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# LigandMPNN benchmark on the exact PDB set used by the UMA-Inverse Gibbs sweep
# (v3-gibbs-K5, max_total_nodes=650).  Reads pdb_id list from that run's
# per_pdb.csv so the two benchmarks are evaluated on identical structures.
#
# Outputs: outputs/benchmark/ligandmpnn-matched-gibbs/

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
LIGANDMPNN_DIR=/private/groups/yehlab/wsobolew/01_software/LigandMPNN

GIBBS_PER_PDB="${PROJ}/outputs/benchmark/v3-gibbs-K0/gibbs_per_pdb.csv"
PDB_DIR="${PROJ}/data/raw/pdb_archive"
OUT_DIR="${PROJ}/outputs/benchmark/ligandmpnn-matched-k0"
INPUTS_DIR="${OUT_DIR}/inputs"
STATS_DIR="${OUT_DIR}/stats"

if [[ ! -f "${GIBBS_PER_PDB}" ]]; then
    echo "!! Gibbs per_pdb.csv not found at ${GIBBS_PER_PDB}" >&2
    echo "   Run the v3-gibbs-K5 benchmark first." >&2
    exit 1
fi

mkdir -p "${INPUTS_DIR}" "${STATS_DIR}"

CHECKPOINT_PATH="${LIGANDMPNN_DIR}/model_params/ligandmpnn_v_32_010_25.pt"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "!! LigandMPNN weights not found at ${CHECKPOINT_PATH}" >&2
    exit 1
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate ligandmpnn_env

# Build pdb_path_multi.json from the Gibbs sweep's PDB IDs
echo ">> Building input JSON from Gibbs per_pdb.csv ($(tail -n +2 ${GIBBS_PER_PDB} | wc -l) PDBs)"
cd "${PROJ}"
python - <<PYEOF
import csv, json
from pathlib import Path

proj    = Path("${PROJ}")
pdb_dir = proj / "data/raw/pdb_archive"
out_dir = proj / "outputs/benchmark/ligandmpnn-matched-k0/inputs"

def resolve(pdb_id):
    pid = pdb_id.lower()
    for p in [pdb_dir / pid[1:3] / f"{pid}.pdb", pdb_dir / f"{pid}.pdb"]:
        if p.exists():
            return p
    return None

pdb_map = {}
missing = []
with open("${GIBBS_PER_PDB}") as f:
    for row in csv.DictReader(f):
        path = resolve(row["pdb_id"])
        if path is not None:
            pdb_map[str(path)] = ""
        else:
            missing.append(row["pdb_id"])

if missing:
    print(f"  warning: {len(missing)} PDB files not found, skipping")

# Split into chunks of 100
paths = list(pdb_map.keys())
chunks_dir = out_dir / "chunks"
chunks_dir.mkdir(parents=True, exist_ok=True)
chunk_files = []
for i in range(0, len(paths), 100):
    chunk = {p: "" for p in paths[i:i+100]}
    cf = chunks_dir / f"chunk_{i//100:03d}.json"
    cf.write_text(json.dumps(chunk, indent=2))
    chunk_files.append(str(cf))

(out_dir / "pdb_path_multi.json").write_text(json.dumps(pdb_map, indent=2))
(chunks_dir / "chunk_list.txt").write_text("\n".join(chunk_files) + "\n")
print(f"resolved {len(pdb_map)} PDBs -> {len(chunk_files)} chunks")
PYEOF

N_CHUNKS=$(wc -l < "${INPUTS_DIR}/chunks/chunk_list.txt")
echo ">> Running LigandMPNN on ${N_CHUNKS} chunks (T=0.1, 1 sample, save_stats)"

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

cd "${PROJ}"
uv run python scripts/paper/benchmark_ligandmpnn_postprocess.py \
    --stats-dir "${OUT_DIR}/stats" \
    --pdb-dir   "${PDB_DIR}" \
    --out-dir   "${OUT_DIR}"

echo ">> Done. Results in ${OUT_DIR}/summary.md"
cat "${OUT_DIR}/summary.md"
