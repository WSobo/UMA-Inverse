#!/bin/bash
#SBATCH --job-name=boltz
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/CZI_ligands/logs/out/boltz_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/CZI_ligands/logs/err/boltz_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Error handling
set -e

# Initialize micromamba for this shell session
eval "$(micromamba shell hook --shell bash)"

# Activate environment
micromamba activate boltz

# Store the original working directory
ORIGINAL_DIR=$(pwd)

# Convert relative paths to absolute paths before changing directory
YAML_FILE=$(realpath "${1:-inputs/yaml/r3no2.yaml}")
OUT_DIR=$(realpath "${2:-outputs/r3no2}")

# Change to boltz2 directory for execution
cd /private/groups/yehlab/wsobolew/02_projects/computational/boltz2

# Set GPU optimizations for better performance
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run boltz with error checking
echo "Starting Boltz prediction..."
echo "Timestamp: $(date)"
echo "YAML file: $YAML_FILE"
echo "Output directory: $OUT_DIR"

# Enable Tensor Core optimization by setting precision
python -c "import torch; torch.set_float32_matmul_precision('medium'); print('Tensor precision optimized')"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

srun boltz predict "$YAML_FILE" \
  --cache /private/groups/yehlab/wsobolew/.boltz_cache_shared \
  --out_dir "$OUT_DIR" \
  --devices 1 \
  --accelerator gpu \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --diffusion_samples 5 \
  --step_scale 1.638 \
  --output_format pdb \
  --sampling_steps_affinity 200 \
  --diffusion_samples_affinity 5 \

echo "Boltz prediction completed at: $(date)"
echo "Results saved to: $OUT_DIR"