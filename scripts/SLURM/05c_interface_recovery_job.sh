#!/bin/bash
#SBATCH --job-name=uma-iface
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/iface_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/iface_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Per-split interface-recovery benchmark job. Always invoked by
# scripts/SLURM/05c_benchmark_interface_recovery.sh which sets IFACE_* env vars.
# For each PDB in --val-json: 10 AR samples at T=0.1, median interface recovery.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> iface-recovery config:"
echo "     ckpt         = $IFACE_CKPT"
echo "     val json     = $IFACE_VAL_JSON"
echo "     pdb dir      = $IFACE_PDB_DIR"
echo "     run name     = $IFACE_RUN_NAME"
echo "     samples/pdb  = $IFACE_NUM_SAMPLES"
echo "     temperature  = $IFACE_TEMPERATURE"
echo "     cutoff Å     = $IFACE_CUTOFF"

uv run python scripts/benchmark_interface_recovery.py \
    --ckpt "$IFACE_CKPT" \
    --val-json "$IFACE_VAL_JSON" \
    --pdb-dir "$IFACE_PDB_DIR" \
    --run-name "$IFACE_RUN_NAME" \
    --num-samples "$IFACE_NUM_SAMPLES" \
    --temperature "$IFACE_TEMPERATURE" \
    --cutoff "$IFACE_CUTOFF"

echo ">> done."
