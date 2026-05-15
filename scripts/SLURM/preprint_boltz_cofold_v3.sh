#!/bin/bash
#SBATCH --job-name=boltz-cofold-v3
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/boltz_cofold_v3_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/boltz_cofold_v3_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Boltz-2 cofold for UMA-Inverse v3 pocket-fixed redesigns.
#
# Re-uses the existing LigandMPNN cofold outputs (v2 run) by symlinking
# them into the v3 cofold tree — LigandMPNN didn't change between v2 and
# v3 runs, so re-cofolding it would just burn GPU time. Only the v3 UMA
# designs are submitted to Boltz-2.
#
# Inputs:
#   outputs/preprint/uma_pocket_fixed_v3/<pdb>/designs.fasta
#     (produced by scripts/SLURM/preprint_uma_pocket_fixed.sh on v3 ckpt)
#   outputs/preprint/cofold/ligandmpnn/   (existing v2-run output, symlinked)
#
# Outputs:
#   outputs/preprint/boltz_inputs/cofold_v3/uma_v3/         YAMLs
#   outputs/preprint/cofold_v3/uma_v3/                     Boltz cofold output
#   outputs/preprint/cofold_v3/ligandmpnn -> ../cofold/ligandmpnn (symlink)
#   outputs/preprint/cofold_v3_metrics.csv                 (after metrics step)

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "${PROJ}"

UMA_DESIGNS="${UMA_DESIGNS:-${PROJ}/outputs/preprint/uma_pocket_fixed_v3}"
INPUT_BASE="${INPUT_BASE:-${PROJ}/outputs/preprint/boltz_inputs/cofold_v3}"
OUT_BASE="${OUT_BASE:-${PROJ}/outputs/preprint/cofold_v3}"
LMPNN_V2_COFOLD="${LMPNN_V2_COFOLD:-${PROJ}/outputs/preprint/cofold/ligandmpnn}"
SELECTION="${SELECTION:-${PROJ}/outputs/preprint/pdb_selection_combined.json}"
CACHE=/private/groups/yehlab/wsobolew/.boltz_cache_shared

mkdir -p "${INPUT_BASE}" "${OUT_BASE}/uma_v3"

# Build Boltz-2 YAMLs from the v3 designs (uses the existing builder
# parameterized via --uma-method-name and --skip-ligandmpnn).
echo ">> Building Boltz-2 YAMLs from v3 UMA designs"
uv run python scripts/paper/build_cofold_yamls.py \
    --selection "${SELECTION}" \
    --uma-dir "${UMA_DESIGNS}" \
    --uma-method-name uma_v3 \
    --out-dir "${INPUT_BASE}" \
    --skip-ligandmpnn

# Reuse the existing LigandMPNN cofold output (it didn't change between
# v2 and v3 runs).
if [[ -d "${LMPNN_V2_COFOLD}" && ! -e "${OUT_BASE}/ligandmpnn" ]]; then
    ln -s "${LMPNN_V2_COFOLD}" "${OUT_BASE}/ligandmpnn"
    echo ">> symlinked ${LMPNN_V2_COFOLD} -> ${OUT_BASE}/ligandmpnn"
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate boltz

cd /private/groups/yehlab/wsobolew/02_projects/computational/boltz2

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python -c "import torch; torch.set_float32_matmul_precision('medium'); print('Tensor precision optimized')"

echo ">> Boltz-2 cofold for UMA-v3 pocket-fixed redesigns"
echo "   inputs: ${INPUT_BASE}/uma_v3"
echo "   output: ${OUT_BASE}/uma_v3"
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

echo ">> Boltz-2 cofold v3 batch complete."
