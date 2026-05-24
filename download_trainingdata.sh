#!/usr/bin/env bash
set -euo pipefail

# UMA-Inverse uses LigandMPNN train/valid/test JSON splits for comparability.
# Place the actual PDB archive under data/raw/pdb_archive/ in RCSB-style folders.
# Example expected path for 1ABC: data/raw/pdb_archive/ab/1ABC.pdb

mkdir -p data/raw/pdb_archive data/processed

echo "Use the LigandMPNN split files:"
echo "  ../LigandMPNN/training/train.json"
echo "  ../LigandMPNN/training/valid.json"
echo "  ../LigandMPNN/training/test_small_molecule.json"

echo "Then run: python scripts/pilot_run.py"
