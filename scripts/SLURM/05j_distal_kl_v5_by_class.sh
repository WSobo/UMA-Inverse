#!/bin/bash
#SBATCH --job-name=uma-distal-byclass
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/distal_byclass_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/distal_byclass_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Distal-residue ligand-conditioning KL shift, reported PER LIGAND CLASS on the
# LigandMPNN test splits (small_molecule / metal / nucleotide) — the v5-aware
# successor to the pooled-val distal_kl run, which predated nucleic-acid support.
# Runs UMA-v5 and LigandMPNN on each split; KL by 5A distance shell.
#
#   sbatch --gres=gpu:1 scripts/SLURM/05j_distal_kl_v5_by_class.sh
#   SKIP_LMPNN=1 ... to run UMA-only.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-checkpoints/pairmixerinv-v5-stage3-nodes384-ddp2/epoch_snapshots/epoch-11.ckpt}"
CONFIG="${CONFIG:-configs/config_v5.yaml}"
NUM_BATCHES="${NUM_BATCHES:-80}"          # >= test_small_molecule(317)/~6 per batch
OUT_BASE="${OUT_BASE:-outputs/preprint/distal_kl/by_class_v5}"
SKIP_LMPNN="${SKIP_LMPNN:-0}"
LMPNN_FLAG=""; [[ "$SKIP_LMPNN" == "1" ]] && LMPNN_FLAG="--skip-ligandmpnn"

[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint not found: $CKPT" >&2; exit 1; }

for cls in small_molecule metal nucleotide; do
    val_json="LigandMPNN/training/test_${cls}.json"
    pdb_dir="data/raw/pdb_archive/test_${cls}"
    out_dir="${OUT_BASE}/${cls}"
    mkdir -p "$out_dir"
    pdb_list="${out_dir}/pdb_list.txt"

    # Build the PDB-id list from the test split JSON.
    uv run python -c "
import json, pathlib
d = json.loads(pathlib.Path('${val_json}').read_text())
ids = list(d.keys()) if isinstance(d, dict) else list(d)
pathlib.Path('${pdb_list}').write_text('\n'.join(ids) + '\n')
print(f'${cls}: {len(ids)} ids -> ${pdb_list}')
"

    echo ">> distal-KL [${cls}]  ckpt=${CKPT}  ($(date))"
    uv run python scripts/paper/distal_kl_shift.py \
        --uma-ckpt "$CKPT" \
        --config "$CONFIG" \
        --pdb-list "$pdb_list" \
        --pdb-dir "$pdb_dir" \
        --pdb-dir data/raw/pdb_archive \
        --out-dir "$out_dir" \
        --num-batches "$NUM_BATCHES" \
        --max-residues 0 \
        --seed 0 \
        $LMPNN_FLAG || { echo "!! ${cls} distal-KL failed — continuing"; continue; }

    # distal_kl_shift.py hard-labels UMA rows "uma-inverse-v3"; the ckpt here is
    # v5, so relabel the UMA rows in the per-class summary (LigandMPNN rows are
    # untouched). Provenance: ckpt path is recorded in manifest.json.
    [[ -f "${out_dir}/distal_kl_summary.csv" ]] && \
        sed -i 's/uma-inverse-v3/uma-inverse-v5/g' "${out_dir}/distal_kl_summary.csv"
    echo ">> [${cls}] done  ($(date))"
done

echo ">> per-class distal-KL complete: ${OUT_BASE}/{small_molecule,metal,nucleotide}/distal_kl_summary.csv"
