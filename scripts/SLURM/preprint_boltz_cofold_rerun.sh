#!/bin/bash
#SBATCH --job-name=boltz-cofold-rerun
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/boltz_cofold_rerun_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/boltz_cofold_rerun_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Re-run cofolds for the 4 extended PDBs (1f0r, 1nl9, 1nwl, 1qb1) whose
# YAMLs previously failed with a numeric-CCD-as-int crash inside Boltz.
# Now CCDs are quoted; 4 PDBs x 2 methods x 5 samples = 40 cofolds total,
# ~2-3h on 1 GPU.

set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate boltz

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
INPUT_BASE="${PROJ}/outputs/preprint/boltz_inputs/cofold_rerun"
OUT_BASE="${PROJ}/outputs/preprint/cofold_rerun"
CACHE=/private/groups/yehlab/wsobolew/.boltz_cache_shared

mkdir -p "${OUT_BASE}/uma_v2" "${OUT_BASE}/ligandmpnn"

cd /private/groups/yehlab/wsobolew/02_projects/computational/boltz2

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python -c "import torch; torch.set_float32_matmul_precision('medium'); print('Tensor precision optimized')"

echo ">> Boltz-2 cofold rerun (UMA, 4 PDBs)"
srun boltz predict "${INPUT_BASE}/uma_v2" \
    --cache "${CACHE}" --out_dir "${OUT_BASE}/uma_v2" \
    --devices 1 --accelerator gpu \
    --recycling_steps 3 --sampling_steps 200 --diffusion_samples 5 \
    --step_scale 1.638 --output_format pdb \
    --sampling_steps_affinity 200 --diffusion_samples_affinity 5

echo ">> Boltz-2 cofold rerun (LigandMPNN, 4 PDBs)"
srun boltz predict "${INPUT_BASE}/ligandmpnn" \
    --cache "${CACHE}" --out_dir "${OUT_BASE}/ligandmpnn" \
    --devices 1 --accelerator gpu \
    --recycling_steps 3 --sampling_steps 200 --diffusion_samples 5 \
    --step_scale 1.638 --output_format pdb \
    --sampling_steps_affinity 200 --diffusion_samples_affinity 5

echo ">> rerun complete."
