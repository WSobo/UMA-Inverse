#!/bin/bash
#SBATCH --job-name=cofold-ext-array
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/cofold_ext_arr_%A_%a.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/cofold_ext_arr_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Stage 3: parallel Boltz-2 cofold of one chunk of new records per array task.
# Each task writes to its own per-chunk out dir (no cross-task races); chunk
# input subdirs are named uma_v5/native/ligandmpnn so boltz_results_<arm> naming
# is preserved for the downstream merge. Empty chunks exit cleanly.

set -e
PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
K_DIR="$PROJ/outputs/preprint/boltz_inputs/cofold_chunks/chunk_${SLURM_ARRAY_TASK_ID}"
OUT="$PROJ/outputs/preprint/cofold_chunks_out/chunk_${SLURM_ARRAY_TASK_ID}"
CACHE=/private/groups/yehlab/wsobolew/.boltz_cache_shared

if [ ! -d "$K_DIR" ]; then
    echo ">> no chunk dir $K_DIR -- nothing to do, exiting"
    exit 0
fi

(
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate boltz
    cd /private/groups/yehlab/wsobolew/02_projects/computational/boltz2
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    for arm in uma_v5 native ligandmpnn; do
        indir="$K_DIR/$arm"
        ls "$indir"/*.yaml >/dev/null 2>&1 || { echo ">> chunk ${SLURM_ARRAY_TASK_ID} arm=$arm empty -- skip"; continue; }
        echo ">> chunk ${SLURM_ARRAY_TASK_ID} cofold arm=$arm  ($(date))"
        boltz predict "$indir" \
            --cache "$CACHE" \
            --out_dir "$OUT/$arm" \
            --devices 1 --accelerator gpu \
            --recycling_steps 3 --sampling_steps 200 \
            --diffusion_samples 5 --step_scale 1.638 \
            --output_format pdb \
            --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
            || echo "!! boltz failed arm=$arm chunk=${SLURM_ARRAY_TASK_ID} -- continuing"
    done
)
echo ">> chunk ${SLURM_ARRAY_TASK_ID} complete  ($(date))"
