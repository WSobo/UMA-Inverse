#!/bin/bash
#SBATCH --job-name=lmpnn-iface-score
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/lmpnn_iface_score_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/lmpnn_iface_score_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Score the LigandMPNN interface designs (from 05q) through the IDENTICAL path
# UMA-Inverse uses: same compute_sidechain_interface_mask, same ctx from
# session.load_structure (so residue_ids/native are identical), same per-PDB
# median over 10 samples, same mean-of-per-PDB-medians headline. The UMA ckpt is
# loaded only to build ctx; the scored sequences are LigandMPNN's. A native-string
# assertion guards alignment. Output dirs ligandmpnn-test_<cls> sit next to the
# UMA v5-test_<cls> dirs so the figures can read both with one code path.

set -e
PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-checkpoints/uma-inverse-v5.ckpt}"
[[ -f "$CKPT" ]] || CKPT=checkpoints/pairmixerinv-v5-stage3-nodes384-ddp2/epoch_snapshots/epoch-11.ckpt
CONFIG="${CONFIG:-configs/config_v5.yaml}"

for cls in small_molecule metal nucleotide; do
    echo ">> scoring LigandMPNN interface recovery [$cls]  ($(date))"
    uv run python scripts/benchmark_interface_recovery.py \
        --ckpt "$CKPT" \
        --config "$CONFIG" \
        --val-json "LigandMPNN/training/test_${cls}.json" \
        --pdb-dir "data/raw/pdb_archive/test_${cls}" \
        --run-name "ligandmpnn-test_${cls}" \
        --out-dir outputs/benchmark/interface_recovery \
        --ligandmpnn-seqs-dir "outputs/benchmark/ligandmpnn_interface/${cls}" \
        --num-samples 10 --cutoff 5.0 --seed 0
done
echo ">> LigandMPNN interface scoring complete  ($(date))"
