"""Compare UMA-Inverse v3 vs LigandMPNN sequence recovery by distance-to-ligand shell.

Reads:
  outputs/benchmark/v3-final/per_position.parquet      UMA teacher-forced
  outputs/benchmark/v3-ar-T0.1/temperature_sweep.csv   UMA AR T=0.1 headline
  outputs/benchmark/ligandmpnn-val2000/per_position.parquet  LigandMPNN AR T=0.1
  outputs/benchmark/ligandmpnn-val2000/summary.json     LigandMPNN headline

Outputs:
  outputs/preprint/shell_recovery_comparison.csv
  outputs/preprint/shell_recovery_comparison.png  (if matplotlib available)
  outputs/preprint/shell_recovery_summary.md

The UMA per_position data uses teacher-forced predictions; LigandMPNN data
uses autoregressive T=0.1 predictions.  Shell recovery comparisons use
teacher-forced UMA (best available per-position data for UMA) vs AR LigandMPNN
and note the caveat.  Once UMA AR per-position data is available (requires
extending the temperature sweep to output per-position parquet), re-run
with --uma-parquet pointing at that file for a fully fair comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shell_comparison")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SHELLS = [
    (0.0,  5.0,  "0–5 Å"),
    (5.0,  10.0, "5–10 Å"),
    (10.0, 15.0, "10–15 Å"),
    (15.0, 20.0, "15–20 Å"),
    (20.0, float("inf"), ">20 Å"),
]


def _shell_recovery(df: pd.DataFrame) -> list[dict]:
    rows = []
    for lo, hi, label in SHELLS:
        mask = (df["distance_to_ligand"] >= lo) & (df["distance_to_ligand"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        correct_col = "correct" if "correct" in sub.columns else None
        if correct_col:
            rec = sub["correct"].mean()
        else:
            rec = (sub["pred_token"] == sub["native_token"]).mean()
        rows.append({
            "shell": label, "lo_A": lo, "hi_A": hi,
            "n_residues": len(sub), "recovery": float(rec),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uma-parquet",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmark/v3-final/per_position.parquet",
        help="UMA per_position.parquet (teacher-forced by default).",
    )
    parser.add_argument(
        "--lmpnn-parquet",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmark/ligandmpnn-val2000/per_position.parquet",
    )
    parser.add_argument(
        "--lmpnn-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmark/ligandmpnn-val2000/summary.json",
    )
    parser.add_argument(
        "--uma-ar-sweep",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmark/v3-ar-T0.1/temperature_sweep.csv",
        help="UMA temperature_sweep.csv at T=0.1 for headline AR recovery.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/preprint",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if not args.uma_parquet.exists():
        raise SystemExit(f"UMA parquet not found: {args.uma_parquet}")
    if not args.lmpnn_parquet.exists():
        raise SystemExit(f"LigandMPNN parquet not found: {args.lmpnn_parquet}\n"
                         "Run preprint_benchmark_ligandmpnn.sh first.")

    uma_df = pd.read_parquet(args.uma_parquet)
    lmp_df = pd.read_parquet(args.lmpnn_parquet)

    # Filter to finite distances (PDBs with ligand atoms resolved)
    uma_df = uma_df[np.isfinite(uma_df["distance_to_ligand"])]
    lmp_df = lmp_df[np.isfinite(lmp_df["distance_to_ligand"])]

    # Filter out X tokens (native_token == 20) — these are excluded in both models
    uma_df = uma_df[uma_df["native_token"] != 20]
    lmp_df = lmp_df[lmp_df["native_token"] != 20]

    # For UMA, use only the non-masked pass (ligand context present)
    if "ligand_context_masked" in uma_df.columns:
        uma_df = uma_df[~uma_df["ligand_context_masked"]]

    # Add correct column to UMA df if not present
    if "correct" not in uma_df.columns:
        uma_df = uma_df.copy()
        uma_df["correct"] = (uma_df["pred_token"] == uma_df["native_token"]).astype(int)

    logger.info("UMA:       %d positions across %d PDBs",
                len(uma_df), uma_df["pdb_id"].nunique())
    logger.info("LigandMPNN: %d positions across %d PDBs",
                len(lmp_df), lmp_df["pdb_id"].nunique())

    # ── Shell recovery ────────────────────────────────────────────────────────
    uma_shells = _shell_recovery(uma_df)
    lmp_shells = _shell_recovery(lmp_df)

    lmp_by_shell = {r["shell"]: r for r in lmp_shells}

    combined_rows = []
    for uma_r in uma_shells:
        lmp_r = lmp_by_shell.get(uma_r["shell"])
        combined_rows.append({
            "shell": uma_r["shell"],
            "lo_A": uma_r["lo_A"],
            "hi_A": uma_r["hi_A"],
            "uma_recovery": uma_r["recovery"],
            "uma_n_residues": uma_r["n_residues"],
            "lmpnn_recovery": lmp_r["recovery"] if lmp_r else float("nan"),
            "lmpnn_n_residues": lmp_r["n_residues"] if lmp_r else 0,
            "delta": uma_r["recovery"] - (lmp_r["recovery"] if lmp_r else float("nan")),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "shell_recovery_comparison.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
        w.writeheader()
        w.writerows(combined_rows)
    logger.info("wrote %s", out_csv)

    # ── Headline numbers ──────────────────────────────────────────────────────
    uma_pooled_tf = float((uma_df["correct"]).mean())
    uma_ar_recovery = None
    if args.uma_ar_sweep.exists():
        ar_df = pd.read_csv(args.uma_ar_sweep)
        t01 = ar_df[ar_df["temperature"].round(2) == 0.1]
        if len(t01) > 0:
            uma_ar_recovery = float(t01["mean_recovery"].iloc[0])

    lmpnn_pooled = None
    if args.lmpnn_summary.exists():
        lmpnn_pooled = json.loads(args.lmpnn_summary.read_text()).get("pooled_recovery")

    # ── Figure ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shells = [r["shell"] for r in combined_rows]
        uma_rec = [r["uma_recovery"] for r in combined_rows]
        lmp_rec = [r["lmpnn_recovery"] for r in combined_rows]
        x = np.arange(len(shells))
        w = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w/2, uma_rec, w, label="UMA-Inverse v3 (teacher-forced)", color="#4C72B0")
        ax.bar(x + w/2, lmp_rec, w, label="LigandMPNN (AR T=0.1)", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(shells, rotation=15, ha="right")
        ax.set_ylabel("Sequence recovery")
        ax.set_ylim(0, 1)
        ax.set_title("Recovery by distance-to-ligand shell\n"
                     "(UMA teacher-forced vs LigandMPNN AR T=0.1)")
        ax.legend()
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        fig.tight_layout()
        fig_path = args.out_dir / "shell_recovery_comparison.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("wrote %s", fig_path)
    except Exception as e:
        logger.warning("figure skipped: %s", e)

    # ── Summary markdown ──────────────────────────────────────────────────────
    table = "\n".join(
        f"| {r['shell']:10s} | {r['uma_recovery']:.3f} ({r['uma_n_residues']:>6d}) "
        f"| {r['lmpnn_recovery']:.3f} ({r['lmpnn_n_residues']:>6d}) "
        f"| {r['delta']:+.3f} |"
        for r in combined_rows
    )
    headline = ""
    if uma_ar_recovery is not None:
        headline += f"- **UMA v3 AR T=0.1 recovery**: {uma_ar_recovery:.4f}\n"
    if lmpnn_pooled is not None:
        headline += f"- **LigandMPNN AR T=0.1 recovery**: {lmpnn_pooled:.4f}\n"
    headline += f"- **UMA v3 teacher-forced recovery**: {uma_pooled_tf:.4f}\n"

    md = f"""# Shell recovery comparison: UMA-Inverse v3 vs LigandMPNN

## Headline recovery (T=0.1 autoregressive — apples-to-apples)

{headline}
> Note: shell table uses UMA teacher-forced per-position data (best available).
> Δ = UMA − LigandMPNN; positive = UMA wins.

## Recovery by distance-to-ligand shell

| Shell      | UMA v3 (TF)           | LigandMPNN (AR)       | Δ      |
|------------|-----------------------|-----------------------|--------|
{table}

## Interpretation

- **Pocket shell (0–5 Å)**: direct ligand-contacting residues — both models see the ligand.
- **Near shell (5–15 Å)**: first coordination sphere of the pocket.
- **Distal shell (>15 Å)**: where PairMixer's all-pairs attention should have an advantage
  over KNN's local message passing.

If UMA wins at >15 Å and ties/loses at <5 Å, that is evidence that dense pair
encoding captures long-range allosteric coupling that KNN-GNN misses.
"""
    (args.out_dir / "shell_recovery_summary.md").write_text(md)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'Shell':12s}  {'UMA (TF)':>10s}  {'LMPNN (AR)':>10s}  {'Δ':>8s}")
    print("-" * 48)
    for r in combined_rows:
        print(f"{r['shell']:12s}  {r['uma_recovery']:>10.4f}  "
              f"{r['lmpnn_recovery']:>10.4f}  {r['delta']:>+8.4f}")
    if uma_ar_recovery is not None:
        print(f"\nUMA AR T=0.1 pooled:     {uma_ar_recovery:.4f}")
    if lmpnn_pooled is not None:
        print(f"LigandMPNN AR T=0.1:     {lmpnn_pooled:.4f}")


if __name__ == "__main__":
    main()
