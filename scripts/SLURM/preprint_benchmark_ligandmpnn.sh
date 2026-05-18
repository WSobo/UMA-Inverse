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

# ── Build pdb_path_multi.json from val.json (resolve actual PDB paths) ────────
echo ">> Resolving ${N_PDBS} PDB paths from val.json"
cd "${PROJ}"
uv run python - <<'PYEOF'
import json, os, sys
from pathlib import Path

proj = Path(os.environ["PROJ"])
val_ids = json.loads((proj / "LigandMPNN/training/valid.json").read_text())
pdb_dir = proj / "data/raw/pdb_archive"
n_pdbs  = int(os.environ.get("N_PDBS", 2000))

def resolve(pdb_id):
    pid = pdb_id.lower()
    for p in [pdb_dir / pid[1:3] / f"{pid}.pdb", pdb_dir / f"{pid}.pdb"]:
        if p.exists():
            return str(p)
    return None

pdb_map = {}
for pdb_id in val_ids[:n_pdbs]:
    path = resolve(pdb_id)
    if path:
        pdb_map[path] = ""
    if len(pdb_map) >= n_pdbs:
        break

out = proj / "outputs/benchmark" / f"ligandmpnn-val{n_pdbs}" / "inputs" / "pdb_path_multi.json"
out.write_text(json.dumps(pdb_map, indent=2))
print(f"resolved {len(pdb_map)} / {n_pdbs} PDB paths -> {out}")
PYEOF

N_RESOLVED=$(python3 -c "import json; d=json.load(open('${INPUTS_DIR}/pdb_path_multi.json')); print(len(d))")
echo ">> Running LigandMPNN on ${N_RESOLVED} PDBs (T=0.1, 1 sample, save_stats)"

eval "$(micromamba shell hook --shell bash)"
micromamba activate ligandmpnn_env

cd "${LIGANDMPNN_DIR}"

python run.py \
    --model_type ligand_mpnn \
    --checkpoint_ligand_mpnn "${CHECKPOINT_PATH}" \
    --pdb_path_multi "${INPUTS_DIR}/pdb_path_multi.json" \
    --out_folder "${OUT_DIR}" \
    --batch_size 1 \
    --number_of_batches 1 \
    --temperature 0.1 \
    --seed 0 \
    --save_stats 1

echo ">> LigandMPNN inference done. Post-processing stats..."

cd "${PROJ}"
uv run python scripts/paper/benchmark_ligandmpnn_postprocess.py \
    --stats-dir "${OUT_DIR}/stats" \
    --pdb-dir   "${PDB_DIR}" \
    --out-dir   "${OUT_DIR}"

echo ">> Done. Results in ${OUT_DIR}/summary.md"
cat "${OUT_DIR}/summary.md"
