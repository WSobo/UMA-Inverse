#!/bin/bash
#SBATCH --job-name=uma-cofold-v5
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_out/cofold_v5_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2/logs/SLURM_err/cofold_v5_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# 3-arm pocket-fixed cofold validation: UMA-v5 / native / LigandMPNN.
#   [1/3] build all three arms' Boltz YAMLs from the pocket-fixed designs (uv)
#   [2/3] Boltz-2 cofold each arm directory (boltz micromamba env)
#   [3/3] compute pocket / ligand-pose / scaffold RMSD + confidence (uv)
# Intended to run after the LigandMPNN pocket-fixed job:
#   sbatch --dependency=afterany:<ligandmpnn_jobid> scripts/SLURM/05k_cofold_v5.sh
# afterany (not afterok) so the UMA-v5 + native arms still cofold even if the
# LigandMPNN arm is unavailable; empty arm dirs are skipped.

set -e
PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse-2
cd "${PROJ}"

INPUT_BASE="${PROJ}/outputs/preprint/boltz_inputs/cofold"
OUT_BASE="${PROJ}/outputs/preprint/cofold"
CACHE=/private/groups/yehlab/wsobolew/.boltz_cache_shared
SAMPLES_PER_PDB="${SAMPLES_PER_PDB:-3}"   # designs cofolded per PDB per method

# ── [1/3] build YAMLs for all three arms (uv venv) ──────────────────────────
echo ">> [1/3] building 3-arm cofold YAMLs  ($(date))"
uv run python scripts/paper/build_cofold_yamls.py \
    --selection outputs/preprint/pdb_selection_combined.json \
    --uma-dir outputs/preprint/uma_pocket_fixed_v5 \
    --uma-method-name uma_v5 \
    --ligandmpnn-dir outputs/preprint/ligandmpnn_pocket_fixed \
    --include-native \
    --samples-per-pdb "${SAMPLES_PER_PDB}"

# ── [2/3] Boltz-2 cofold each arm (boltz env, isolated subshell) ─────────────
echo ">> [2/3] Boltz-2 cofold  ($(date))"
mkdir -p "${OUT_BASE}/uma_v5" "${OUT_BASE}/native" "${OUT_BASE}/ligandmpnn"
(
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate boltz
    cd /private/groups/yehlab/wsobolew/02_projects/computational/boltz2
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    python -c "import torch; torch.set_float32_matmul_precision('medium')"
    for arm in uma_v5 native ligandmpnn; do
        in_dir="${INPUT_BASE}/${arm}"
        if ! ls "${in_dir}"/*.yaml >/dev/null 2>&1; then
            echo "!! no YAMLs in ${in_dir} -- skipping arm ${arm}"
            continue
        fi
        echo ">> cofold arm=${arm}  ($(date))"
        boltz predict "${in_dir}" \
            --cache "${CACHE}" \
            --out_dir "${OUT_BASE}/${arm}" \
            --devices 1 --accelerator gpu \
            --recycling_steps 3 --sampling_steps 200 \
            --diffusion_samples 5 --step_scale 1.638 \
            --output_format pdb \
            --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
            || echo "!! boltz cofold failed for ${arm} -- continuing"
    done
)

# ── [3/3] RMSD + confidence metrics (uv venv); non-fatal so cofold outputs
#         survive a metrics bug (re-runnable by hand) ────────────────────────
echo ">> [3/3] cofold metrics  ($(date))"
uv run python scripts/paper/cofold_metrics.py \
    --sampling-record outputs/preprint/boltz_inputs/cofold/sampling_record.json \
    --cofold-base outputs/preprint/cofold \
    --selection outputs/preprint/pdb_selection_combined.json \
    --out outputs/preprint/cofold_metrics_v5.csv \
    || echo "!! cofold_metrics failed -- cofold outputs are on disk under ${OUT_BASE}; rerun metrics by hand"

echo ">> 3-arm cofold pipeline complete  ($(date))"
