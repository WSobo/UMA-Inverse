#!/bin/bash
#SBATCH --job-name=uma-dl-test
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/download_test_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/download_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Download the three LigandMPNN benchmark test splits into per-class
# subdirectories so each archive is self-contained and easy to manage:
#
#   data/raw/pdb_archive/test_metal/<xx>/<id>.pdb           (~83 PDBs)
#   data/raw/pdb_archive/test_nucleotide/<xx>/<id>.pdb      (~76 PDBs)
#   data/raw/pdb_archive/test_small_molecule/<xx>/<id>.pdb  (~292 PDBs)
#
# The ~30 PDBs that appear in multiple splits get downloaded once per
# subdir — small overhead, big gain in organisational clarity.

set -e
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse

echo ">> downloading LigandMPNN test splits into per-class subdirs under data/raw/pdb_archive/"
echo "   host: $SLURM_NODELIST   timestamp: $(date)"

uv run python scripts/download_json_pdbs.py \
    --json-files test_metal.json test_nucleotide.json test_small_molecule.json \
    --split-dirs \
    --workers 8

echo ">> done at $(date)"
