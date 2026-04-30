#!/bin/bash
#SBATCH --job-name=preprint-boltz-cofold
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/boltz_cofold_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/boltz_cofold_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Phase B1: Boltz-2 cofold of pocket-fixed redesigns.
# Reads YAML inputs from outputs/preprint/boltz_inputs/cofold/{uma_v2,ligandmpnn}/
# and runs them in two batches (one per method directory). Boltz-2 supports
# directory-mode invocation, so each batch loads the model once and processes
# all 100 YAMLs sequentially.
#
# Environment / flags match scripts/SLURM/run_boltz_example.sh (the user's
# existing template from the CZI_ligands project).

set -e

# Initialize micromamba for this shell session
eval "$(micromamba shell hook --shell bash)"
micromamba activate boltz

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
INPUT_BASE="${PROJ}/outputs/preprint/boltz_inputs/cofold"
OUT_BASE="${PROJ}/outputs/preprint/cofold"
CACHE=/private/groups/yehlab/wsobolew/.boltz_cache_shared

mkdir -p "${OUT_BASE}/uma_v2" "${OUT_BASE}/ligandmpnn"

cd /private/groups/yehlab/wsobolew/02_projects/computational/boltz2

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python -c "import torch; torch.set_float32_matmul_precision('medium'); print('Tensor precision optimized')"

# ── UMA cofolds ─────────────────────────────────────────────────────────────
echo ">> Boltz-2 cofold for UMA pocket-fixed redesigns"
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

# ── LigandMPNN cofolds ──────────────────────────────────────────────────────
echo ">> Boltz-2 cofold for LigandMPNN pocket-fixed redesigns"
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

echo ">> Boltz-2 cofold batch complete."
