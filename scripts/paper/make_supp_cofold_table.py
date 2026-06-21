"""Supplementary Table S1: per-PDB Boltz-2 cofold metrics.

Aggregates the per-sequence cofold metrics (outputs/preprint/cofold_metrics_ext2.csv,
5 designed sequences per PDB per method, each carrying its best-of-5-diffusion value)
to one row per (pdb_id, method). For each metric we report the mean and standard
deviation across the designed sequences of the per-sequence best-of-5-diffusion value;
affinity fields have no diffusion-best variant and are averaged directly.

Outputs:
    outputs/preprint/supp_table_s1_cofold_per_pdb.csv   (released data file)
Also prints a method-level summary (mean +/- sd per metric) for the appendix table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# (source column, output base name) -- best-of-5-diffusion metrics.
BEST_METRICS = [
    ("confidence_score_best", "confidence_score"),
    ("iptm_best", "iptm"),
    ("ligand_iptm_best", "ligand_iptm"),
    ("ptm_best", "ptm"),
    ("complex_plddt_best", "complex_plddt"),
    ("complex_iplddt_best", "complex_iplddt"),
    ("pocket_calpha_rmsd_best", "pocket_calpha_rmsd"),
    ("ligand_rmsd_best", "ligand_rmsd"),
    ("scaffold_rmsd_best", "scaffold_rmsd"),
]
# Fields with no diffusion-best variant -- averaged directly across sequences.
DIRECT_METRICS = [
    ("affinity_pred_value", "affinity_pred_value"),
    ("affinity_probability_binary", "affinity_probability_binary"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics_ext2.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "supp_table_s1_cofold_per_pdb.csv",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        raise SystemExit(f"missing {args.metrics}")

    df = pd.read_csv(args.metrics)

    grouped = df.groupby(["pdb_id", "kind", "method"], sort=True)

    out_rows = []
    for (pdb_id, kind, method), g in grouped:
        row = {
            "pdb_id": pdb_id,
            "kind": kind,
            "method": method,
            "n_sequences": len(g),
        }
        for src, base in BEST_METRICS + DIRECT_METRICS:
            row[f"{base}_mean"] = g[src].mean()
            row[f"{base}_sd"] = g[src].std(ddof=0)
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, float_format="%.4f")
    print(f"wrote {args.out}  ({len(out_df)} rows)")

    # Method-level summary for the appendix table (mean +/- sd across PDBs of the
    # per-PDB mean value).
    print("\n=== method-level summary (mean +/- sd across PDBs) ===")
    for method in sorted(out_df["method"].unique()):
        sub = out_df[out_df["method"] == method]
        print(f"\n[{method}]  N={len(sub)} PDBs")
        for _, base in BEST_METRICS + DIRECT_METRICS:
            col = f"{base}_mean"
            print(f"  {base:28s} {sub[col].mean():8.3f} +/- {sub[col].std(ddof=0):.3f}")


if __name__ == "__main__":
    main()
