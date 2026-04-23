#!/bin/bash
# Submit one LigandMPNN-style interface-recovery benchmark per test split.
# Each job calls scripts/benchmark_interface_recovery.py with its per-class
# PDB archive and writes outputs under
# outputs/benchmark/interface_recovery/ep11-test_<class>/.
#
# Protocol (matches Dauparas et al.): for every PDB, generate 10 autoregressive
# samples at T=0.1 with random decoding order, compute recovery restricted to
# sidechain-interface residues (≤5 Å of any nonprotein heavy atom), take the
# median across the 10 samples → one scalar per PDB. Headline in summary.json
# is the mean of per-PDB medians.
#
# Usage:
#   bash scripts/SLURM/05c_benchmark_interface_recovery.sh
#   CKPT=checkpoints/last.ckpt bash scripts/SLURM/05c_benchmark_interface_recovery.sh

set -e
cd "$(dirname "$0")/../.."

CKPT="${CKPT:-checkpoints/epoch_snapshots/epoch-epoch=11.ckpt}"
PDB_ROOT="${PDB_ROOT:-data/raw/pdb_archive}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.1}"
CUTOFF="${CUTOFF:-5.0}"

if [[ ! -f "$CKPT" ]]; then
    echo "!! checkpoint not found: $CKPT" >&2
    exit 1
fi

echo ">> interface-recovery benchmark"
echo ">>   ckpt         : $CKPT"
echo ">>   pdb archive  : $PDB_ROOT"
echo ">>   samples/pdb  : $NUM_SAMPLES   T=$TEMPERATURE   cutoff=${CUTOFF}Å"

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
        --export=ALL,IFACE_CKPT="$CKPT",IFACE_VAL_JSON="$val_json",IFACE_PDB_DIR="$pdb_dir",IFACE_RUN_NAME="$run_name",IFACE_NUM_SAMPLES="$NUM_SAMPLES",IFACE_TEMPERATURE="$TEMPERATURE",IFACE_CUTOFF="$CUTOFF" \
        scripts/SLURM/05c_interface_recovery_job.sh
done

echo ">> three jobs submitted. Watch with: squeue -u \$USER"
