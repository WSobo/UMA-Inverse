#!/bin/bash
# Submit one benchmark sbatch per LigandMPNN test split (metal, nucleotide,
# small_molecule). Reuses the existing scripts/SLURM/05_benchmark.sh wrapper
# via its BENCH_* env-var overrides.
#
# Usage:
#   bash scripts/SLURM/05b_benchmark_all_tests.sh
#   CKPT=checkpoints/last.ckpt bash scripts/SLURM/05b_benchmark_all_tests.sh

set -e
cd "$(dirname "$0")/../.."

CKPT="${CKPT:-checkpoints/epoch_snapshots/epoch-epoch=11.ckpt}"
PDB_ROOT="${PDB_ROOT:-data/raw/pdb_archive}"

if [[ ! -f "$CKPT" ]]; then
    echo "!! checkpoint not found: $CKPT" >&2
    exit 1
fi

echo ">> benchmarking ckpt: $CKPT"
echo ">> pdb archive root: $PDB_ROOT"

for cls in metal nucleotide small_molecule; do
    run_name="ep11-test_${cls}"
    val_json="LigandMPNN/training/test_${cls}.json"
    pdb_dir="${PDB_ROOT}/test_${cls}"

    if [[ ! -d "$pdb_dir" ]]; then
        echo "!! pdb dir missing: $pdb_dir (run scripts/SLURM/01b_download_test_splits.sh first)" >&2
        exit 1
    fi

    echo ">> submitting $run_name  (val=$val_json, pdb_dir=$pdb_dir)"
    sbatch \
        --export=ALL,BENCH_CKPT="$CKPT",BENCH_VAL_JSON="$val_json",BENCH_PDB_DIR="$pdb_dir",BENCH_RUN_NAME="$run_name",BENCH_N=all \
        scripts/SLURM/05_benchmark.sh
done

echo ">> three jobs submitted. Watch with: squeue -u \$USER"
