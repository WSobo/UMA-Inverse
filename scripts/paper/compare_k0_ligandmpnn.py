"""Apples-to-apples K=0 sampled recovery: UMA-Inverse vs LigandMPNN.

Both CSVs must have a ``pdb_id`` column and a ``mean_recovery`` /
``recovery`` column respectively.  The intersection of PDB IDs is used so
the comparison is on identical structures.

Usage:
    uv run python scripts/paper/compare_k0_ligandmpnn.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uma-k0",
        default="outputs/benchmark/v3-gibbs-K0/gibbs_per_pdb.csv",
        help="UMA-Inverse gibbs_per_pdb.csv (K=0 rows only).",
    )
    parser.add_argument(
        "--ligandmpnn",
        default="outputs/benchmark/ligandmpnn-matched-k0/per_pdb.csv",
        help="LigandMPNN per_pdb.csv from the matched run.",
    )
    args = parser.parse_args()

    uma_df = pd.read_csv(args.uma_k0)
    uma_df = uma_df[uma_df["num_iterations"] == 0][["pdb_id", "mean_recovery"]]
    uma_df = uma_df.rename(columns={"mean_recovery": "uma_recovery"})

    lmpnn_df = pd.read_csv(args.ligandmpnn)[["pdb_id", "recovery"]]
    lmpnn_df = lmpnn_df.rename(columns={"recovery": "lmpnn_recovery"})

    merged = uma_df.merge(lmpnn_df, on="pdb_id")
    n = len(merged)

    if n == 0:
        print("ERROR: no overlapping PDB IDs — check file paths")
        return

    uma_mean = merged["uma_recovery"].mean()
    lmpnn_mean = merged["lmpnn_recovery"].mean()
    delta = uma_mean - lmpnn_mean

    t_stat, p_value = stats.ttest_rel(merged["uma_recovery"], merged["lmpnn_recovery"])

    print(f"Matched PDBs:          {n}")
    print(f"UMA-Inverse K=0:       {uma_mean:.4f} ± {merged['uma_recovery'].std():.4f}")
    print(f"LigandMPNN (T=0.1 AR): {lmpnn_mean:.4f} ± {merged['lmpnn_recovery'].std():.4f}")
    print(f"Delta (UMA - LMPNN):   {delta:+.4f}")
    print(f"Paired t-test:         t={t_stat:.3f}, p={p_value:.4g}")

    out = Path("outputs/benchmark/k0_vs_ligandmpnn.csv")
    merged.to_csv(out, index=False)
    print(f"\nPer-PDB table: {out}")


if __name__ == "__main__":
    main()
