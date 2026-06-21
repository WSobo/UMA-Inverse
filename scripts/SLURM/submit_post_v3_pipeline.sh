#!/bin/bash
# Pre-submit the full post-stage-3 pipeline with SLURM dependencies on the
# v3 stage-3 training job. Runs end-to-end automatically once stage 3
# finishes.
#
# Job graph:
#
#   stage3 ──┬─→ benchmark
#            ├─→ pocket-fixed-v3 ──→ cofold-v3 ──→ cofold-metrics-v3
#            ├─→ distal-KL-mechanism
#            ├─→ distal-KL-outcome
#            ├─→ distogram-probe       (v3 retro: encoder geometry diagnostic)
#            └─→ wallclock probe
#
# All branches use afterok:<stage3_id>; the cofold sub-chain uses
# afterok:<pocket_fixed_id> and afterok:<cofold_id>. If stage 3 fails,
# every dependent job is auto-cancelled by SLURM.
#
# Usage:
#   bash scripts/SLURM/submit_post_v3_pipeline.sh                   # auto-detect stage 3 job
#   bash scripts/SLURM/submit_post_v3_pipeline.sh --after 33056844  # specify explicitly
#   bash scripts/SLURM/submit_post_v3_pipeline.sh --dry-run         # print, don't submit
#
# Each branch can be skipped with --skip-<branch>:
#   --skip-benchmark, --skip-pocket-fixed, --skip-distal-kl,
#   --skip-cofold, --skip-wallclock, --skip-distogram-probe

set -e

PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "${PROJ}"

STAGE3_JOB=""
DRY_RUN=0
SKIP_BENCHMARK=0
SKIP_POCKET_FIXED=0
SKIP_DISTAL_KL=0
SKIP_COFOLD=0
SKIP_WALLCLOCK=0
SKIP_DISTOGRAM_PROBE=0

# Defaults — paths used by every job
V3_CKPT="${V3_CKPT:-checkpoints/uma-inverse-v3.ckpt}"
V3_CONFIG="${V3_CONFIG:-configs/old_configs/config_v3.yaml}"
SELECTION="${SELECTION:-outputs/preprint/pdb_selection_combined.json}"
POCKET_FIXED_OUT="${POCKET_FIXED_OUT:-outputs/preprint/uma_pocket_fixed_v3}"
WALLCLOCK_PDB="${WALLCLOCK_PDB:-data/raw/pdb_archive/wg/2wgj.pdb}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --after) STAGE3_JOB="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        --skip-benchmark) SKIP_BENCHMARK=1; shift;;
        --skip-pocket-fixed) SKIP_POCKET_FIXED=1; shift;;
        --skip-distal-kl) SKIP_DISTAL_KL=1; shift;;
        --skip-cofold) SKIP_COFOLD=1; shift;;
        --skip-wallclock) SKIP_WALLCLOCK=1; shift;;
        --skip-distogram-probe) SKIP_DISTOGRAM_PROBE=1; shift;;
        -h|--help) sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

# Auto-detect stage 3 job if not given. Look for jobs named
# "uma-inv-v3-stage3-ddp8" in the current squeue.
if [[ -z "${STAGE3_JOB}" ]]; then
    STAGE3_JOB=$(squeue -u "$USER" -h -o "%i %j" | awk '$2 == "uma-inv-v3-stage3-ddp8" {print $1; exit}')
fi
if [[ -z "${STAGE3_JOB}" ]]; then
    echo "ERROR: could not find stage-3 job (name uma-inv-v3-stage3-ddp8). Pass --after <jobid>." >&2
    exit 1
fi

echo "Stage 3 job: ${STAGE3_JOB}"
echo "V3 ckpt:     ${V3_CKPT}"
echo "V3 config:   ${V3_CONFIG}"
echo "Selection:   ${SELECTION}"
echo

# sbatch wrapper that respects --dry-run. Stdout is the job id only (so
# callers can capture via $(...)); the human-readable command goes to
# stderr in dry-run mode.
_DRY_COUNTER=0
_sbatch() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        _DRY_COUNTER=$((_DRY_COUNTER + 1))
        echo "[dry-run] sbatch $*" >&2
        echo "DRY${_DRY_COUNTER}"
        return 0
    fi
    sbatch --parsable "$@"
}

# ── Branch 1: benchmark (parity table, near-ligand, length-stratified) ───────
BENCH_ID=""
if [[ "${SKIP_BENCHMARK}" -eq 0 ]]; then
    BENCH_ID=$(_sbatch \
        --dependency=afterok:${STAGE3_JOB} \
        --kill-on-invalid-dep=yes \
        --export=ALL,BENCH_CKPT=${V3_CKPT},BENCH_CONFIG=${V3_CONFIG},BENCH_N=all,BENCH_RUN_NAME=v3-final \
        scripts/SLURM/05_benchmark.sh)
    echo "  benchmark:                ${BENCH_ID}"
fi

# ── Branch 2: pocket-fixed redesign + cofold + metrics ───────────────────────
POCKET_ID=""
COFOLD_ID=""
METRICS_ID=""
if [[ "${SKIP_POCKET_FIXED}" -eq 0 ]]; then
    POCKET_ID=$(_sbatch \
        --dependency=afterok:${STAGE3_JOB} \
        --kill-on-invalid-dep=yes \
        scripts/SLURM/preprint_uma_pocket_fixed.sh \
        --ckpt "${V3_CKPT}" \
        --config "${V3_CONFIG}" \
        --selection "${SELECTION}" \
        --out-dir "${POCKET_FIXED_OUT}")
    echo "  pocket-fixed-v3:          ${POCKET_ID}"

    if [[ "${SKIP_COFOLD}" -eq 0 ]]; then
        COFOLD_ID=$(_sbatch \
            --dependency=afterok:${POCKET_ID} \
            --kill-on-invalid-dep=yes \
            --export=ALL,UMA_DESIGNS=${POCKET_FIXED_OUT},SELECTION=${SELECTION} \
            scripts/SLURM/preprint_boltz_cofold_v3.sh)
        echo "  boltz-cofold-v3:          ${COFOLD_ID}"

        METRICS_ID=$(_sbatch \
            --dependency=afterok:${COFOLD_ID} \
            --kill-on-invalid-dep=yes \
            --export=ALL,SELECTION=${SELECTION} \
            scripts/SLURM/preprint_cofold_metrics_v3.sh)
        echo "  cofold-metrics-v3:        ${METRICS_ID}"
    fi
fi

# ── Branch 3: distal-KL mechanism (~200 val PDBs) ────────────────────────────
KL_MECH_ID=""
KL_OUT_ID=""
if [[ "${SKIP_DISTAL_KL}" -eq 0 ]]; then
    KL_MECH_ID=$(_sbatch \
        --dependency=afterok:${STAGE3_JOB} \
        --kill-on-invalid-dep=yes \
        --export=ALL,KL_CKPT=${V3_CKPT},KL_MODE=mechanism,KL_N=200 \
        scripts/SLURM/preprint_distal_kl_shift.sh)
    echo "  distal-kl-mechanism:      ${KL_MECH_ID}"

    KL_OUT_ID=$(_sbatch \
        --dependency=afterok:${STAGE3_JOB} \
        --kill-on-invalid-dep=yes \
        --export=ALL,KL_CKPT=${V3_CKPT},KL_MODE=outcome \
        scripts/SLURM/preprint_distal_kl_shift.sh)
    echo "  distal-kl-outcome:        ${KL_OUT_ID}"
fi

# ── Branch 4: wallclock probe ────────────────────────────────────────────────
WALL_ID=""
if [[ "${SKIP_WALLCLOCK}" -eq 0 ]]; then
    # Inline a tiny wrapper to invoke the wallclock script with the v3 ckpt.
    # Cleaner than a separate SLURM file for a 10-minute one-shot.
    WALL_ID=$(_sbatch \
        --dependency=afterok:${STAGE3_JOB} \
        --kill-on-invalid-dep=yes \
        --job-name=v3-wallclock \
        --time=01:00:00 --mem=16G --cpus-per-task=4 \
        --partition=gpu --gres=gpu:A5500:1 \
        --output=logs/SLURM_out/wallclock_%j.out \
        --error=logs/SLURM_err/wallclock_%j.err \
        --wrap="cd ${PROJ} && uv run python scripts/probe_inference_wallclock.py \
            --pdb ${WALLCLOCK_PDB} \
            --uma-ckpt ${V3_CKPT} \
            --config ${V3_CONFIG} \
            --n-warmup 5 --n-trials 50")
    echo "  wallclock-probe:          ${WALL_ID}"
fi

# ── Branch 5: distogram probe (v3 retro — encoder geometry diagnostic) ───────
PROBE_ID=""
if [[ "${SKIP_DISTOGRAM_PROBE}" -eq 0 ]]; then
    PROBE_ID=$(_sbatch \
        --dependency=afterok:${STAGE3_JOB} \
        --kill-on-invalid-dep=yes \
        --export=ALL,PROBE_CKPT=${V3_CKPT},PROBE_CONFIG=${V3_CONFIG},PROBE_N=400 \
        scripts/SLURM/preprint_distogram_probe.sh)
    echo "  distogram-probe:          ${PROBE_ID}"
fi

echo
echo "All downstream jobs queued behind stage 3 (${STAGE3_JOB})."
echo "Inspect with: squeue -u \$USER"
