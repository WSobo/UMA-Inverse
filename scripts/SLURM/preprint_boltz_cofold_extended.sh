#!/bin/bash
#SBATCH --job-name=preprint-boltz-cofold-ext
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/boltz_cofold_ext_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/boltz_cofold_ext_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Phase A scale-up: Boltz-2 cofold of pocket-fixed redesigns on the 15
# additional small-mol PDBs from outputs/preprint/pdb_selection_extended.json.
# 15 PDBs x 2 methods x 5 samples = 150 cofolds. ~6-8h wall on 1 GPU.
#
# Two-step:
#   1. Build cofold YAMLs from the extended selection (CPU, seconds; uses uv
#      env). Written under outputs/preprint/boltz_inputs/cofold_extended/.
#   2. Run Boltz-2 in directory-mode for each method (boltz env). One model
#      load per method, sequential per-input prediction.

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
EXT_SELECTION="${PROJ}/outputs/preprint/pdb_selection_extended.json"
INPUT_BASE="${PROJ}/outputs/preprint/boltz_inputs/cofold_extended"
OUT_BASE="${PROJ}/outputs/preprint/cofold_extended"
CACHE=/private/groups/yehlab/wsobolew/.boltz_cache_shared

# ── 1. Build cofold YAMLs (uv env) ──────────────────────────────────────────
echo ">> [1/2] Building cofold YAMLs for extended selection"
cd "${PROJ}"
uv run python scripts/preprint/build_cofold_yamls.py \
    --selection "${EXT_SELECTION}" \
    --out-dir "${INPUT_BASE}" 2>&1 | tail -20

# ── 2. Boltz-2 cofold (boltz env) ───────────────────────────────────────────
eval "$(micromamba shell hook --shell bash)"
micromamba activate boltz

mkdir -p "${OUT_BASE}/uma_v2" "${OUT_BASE}/ligandmpnn"

cd /private/groups/yehlab/wsobolew/02_projects/computational/boltz2

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python -c "import torch; torch.set_float32_matmul_precision('medium'); print('Tensor precision optimized')"

echo ">> [2/2a] Boltz-2 cofold for UMA pocket-fixed redesigns (extended)"
echo "   inputs: ${INPUT_BASE}/uma_v2"
echo "   output: ${OUT_BASE}/uma_v2"
srun boltz predict "${INPUT_BASE}/uma_v2" \
    --cache "${CACHE}" \
    --out_dir "${OUT_BASE}/uma_v2" \
    --devices 1 \
    --accelerator gpu \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 5 \
    --step_scale 1.638 \
    --output_format pdb \
    --sampling_steps_affinity 200 \
    --diffusion_samples_affinity 5

echo ">> [2/2b] Boltz-2 cofold for LigandMPNN pocket-fixed redesigns (extended)"
echo "   inputs: ${INPUT_BASE}/ligandmpnn"
echo "   output: ${OUT_BASE}/ligandmpnn"
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

echo ">> Boltz-2 cofold (extended) complete."
