#!/bin/bash
#SBATCH --job-name=boltz-cofold-ext2
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/boltz_cofold_ext2_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/boltz_cofold_ext2_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Boltz-2 cofold for the ext2 expansion (325 new cofolds):
#   - UMA v3 ext2:        25 PDBs x 5 samples = 125 cofolds
#   - LigandMPNN ext1+2:  40 PDBs x 5 samples = 200 cofolds
#
# Requires:
#   preprint_uma_pocket_fixed_ext2.sh  (UMA designs in uma_pocket_fixed_v3/)
#   preprint_ligandmpnn_pocket_fixed_ext2.sh (LMPNN seqs in ligandmpnn_pocket_fixed/)
#
# Outputs:
#   outputs/preprint/boltz_inputs/cofold_v3_ext2/{uma_v3,ligandmpnn}/  YAMLs
#   outputs/preprint/cofold_v3_ext2/{uma_v3,ligandmpnn}/               Boltz results

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "${PROJ}"

INPUT_BASE="${INPUT_BASE:-${PROJ}/outputs/preprint/boltz_inputs/cofold_v3_ext2}"
OUT_BASE="${OUT_BASE:-${PROJ}/outputs/preprint/cofold_v3_ext2}"
UMA_DESIGNS="${UMA_DESIGNS:-${PROJ}/outputs/preprint/uma_pocket_fixed_v3}"
LMPNN_SEQS="${LMPNN_SEQS:-${PROJ}/outputs/preprint/ligandmpnn_pocket_fixed}"
CACHE=/private/groups/yehlab/wsobolew/.boltz_cache_shared

mkdir -p "${INPUT_BASE}" "${OUT_BASE}/uma_v3" "${OUT_BASE}/ligandmpnn"

# ── Build YAMLs ───────────────────────────────────────────────────────────────

echo ">> Building Boltz-2 YAMLs: UMA v3 ext2 (25 PDBs)"
uv run python scripts/paper/build_cofold_yamls.py \
    --selection outputs/preprint/pdb_selection_extended2.json \
    --uma-dir "${UMA_DESIGNS}" \
    --uma-method-name uma_v3 \
    --skip-ligandmpnn \
    --out-dir "${INPUT_BASE}"

echo ">> Building Boltz-2 YAMLs: LigandMPNN ext1+ext2 (40 PDBs)"
uv run python scripts/paper/build_cofold_yamls.py \
    --selection outputs/preprint/pdb_selection_lmpnn_todo.json \
    --ligandmpnn-dir "${LMPNN_SEQS}" \
    --skip-uma \
    --out-dir "${INPUT_BASE}"

echo ">> YAML counts:"
echo "   uma_v3:     $(ls ${INPUT_BASE}/uma_v3/*.yaml 2>/dev/null | wc -l) YAMLs"
echo "   ligandmpnn: $(ls ${INPUT_BASE}/ligandmpnn/*.yaml 2>/dev/null | wc -l) YAMLs"

# ── Boltz-2 cofold ────────────────────────────────────────────────────────────

eval "$(micromamba shell hook --shell bash)"
micromamba activate boltz

cd /private/groups/yehlab/wsobolew/02_projects/computational/boltz2

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo ">> Boltz-2: UMA v3 ext2 (125 cofolds)"
srun boltz predict "${INPUT_BASE}/uma_v3" \
    --cache "${CACHE}" \
    --out_dir "${OUT_BASE}/uma_v3" \
    --devices 1 \
    --accelerator gpu \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 5 \
    --step_scale 1.638 \
    --output_format pdb \
    --sampling_steps_affinity 200 \
    --diffusion_samples_affinity 5

echo ">> Boltz-2: LigandMPNN ext1+ext2 (200 cofolds)"
srun boltz predict "${INPUT_BASE}/ligandmpnn" \
    --cache "${CACHE}" \
    --out_dir "${OUT_BASE}/ligandmpnn" \
    --devices 1 \
    --accelerator gpu \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 5 \
    --step_scale 1.638 \
    --output_format pdb \
    --sampling_steps_affinity 200 \
    --diffusion_samples_affinity 5

echo ">> Boltz-2 cofold ext2 batch complete."
