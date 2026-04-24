#!/bin/bash
#SBATCH --job-name=umainv-v2smoke
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/v2smoke_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/v2smoke_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=medium
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# CPU-only smoke test for UMA-Inverse v2 featurizer phases. Exercises the v1
# path and each opt-in v2 flag on a small fixture PDB, asserts forward-pass
# plumbing and strict-load state_dict incompatibility. Re-run after every
# phase commit on the v2-element-embedding branch.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> v2 smoke test on host $SLURM_NODELIST at $(date)"
uv run python scripts/smoke_test_v2.py
echo ">> v2 smoke test finished at $(date)"
