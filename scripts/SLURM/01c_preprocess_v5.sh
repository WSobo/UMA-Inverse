#!/bin/bash
#SBATCH --job-name=uma-preprocess-v5
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/preprocess_v5_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/preprocess_v5_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=medium
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# ── v5 cache build (CPU-only) ────────────────────────────────────────────────
# Re-parses the raw PDB archive into data/processed_v5/ with the v5 data delta:
#   • DNA/RNA ATOM records routed into the ligand atom pool (parser-level,
#     automatic on this branch — the v4 cache dropped them)
#   • ligand_context_atoms = 50 (vs 25 in v4) so NA / large cofactors aren't
#     truncated
# Chemistry / bond topology are OFF (dropped from v5 scope). Same union-cache
# schema as the v4 builder otherwise, so featurizer/anchor/angle flags remain
# pure load-time selections.
#
# Runs on the CPU partition (phoenix-[06-24]); the GPU nodes (phoenix-[00-05])
# stay free for Run A / Control / KL / benchmark jobs. Writes a NEW directory,
# so the data/processed/ cache those GPU jobs read is untouched.
#
# Expected: pure-NA structures (no protein) and the ~1.5% missing IDs fail with
# "No protein residues found" / "pdb_not_found" — benign, same as v4. Failed IDs
# are written to logs/preprocess_v5_failures.txt.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

WORKERS="${PREPROC_WORKERS:-$SLURM_CPUS_PER_TASK}"
OUT_DIR="${PREPROC_OUT_DIR:-data/processed_v5}"

echo ">> preprocess_v5: out_dir=$OUT_DIR  workers=$WORKERS  ($(date))"
echo ">> filesystem headroom:"; df -h -P "$PWD" | tail -1

# --workers pinned to the SLURM allocation: the script's default is
# os.cpu_count() (the FULL node), which would oversubscribe a 32-core slice.
uv run python scripts/preprocess_v5.py \
    --out_dir "$OUT_DIR" \
    --ligand_atoms 50 \
    --max_nodes 1024 \
    --cutoff 8.0 \
    --workers "$WORKERS"

echo ">> done. cache files: $(find "$OUT_DIR" -name '*.pt' | wc -l)  ($(date))"
echo ">> failures (if any): logs/preprocess_v5_failures.txt"
