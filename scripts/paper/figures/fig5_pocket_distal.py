"""Figure 5: pocket-fixed redesign distal-position metrics.

Two-row figure (one per metric):
  Row 1: distal sequence diversity (mean pairwise Hamming distance at distal
         positions). UMA vs LigandMPNN per PDB, paired plot.
  Row 2: distal recovery vs distal diversity scatter, color-coded by
         method, marker shape by split (metal/small_molecule).

Source:
    outputs/preprint/pocket_fixed_summary.csv

The central claim of the preprint is operationalized here -- whether
UMA-Inverse's dense attention enables a measurably distinct distal-redesign
behavior compared to LigandMPNN's KNN-message-passing.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            for k, v in list(row.items()):
                if k in {"pdb_id", "kind", "method"}:
                    continue
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pocket_fixed_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()

    if not args.summary.exists():
        raise SystemExit(
            f"missing {args.summary}. Run scripts/paper/compute_pocket_fixed_metrics.py "
            "after Phase A's UMA + LigandMPNN designs are produced."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load(args.summary)
    by_pdb: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_pdb[r["pdb_id"]][r["method"]] = r
    paired = [pid for pid, d in by_pdb.items() if "uma_v2" in d and "ligandmpnn" in d]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # ── Row 1: distal diversity, paired bar / connected points by PDB ──────
    for col, split_filter, split_name in ((0, "small_molecule", "Small molecule"),
                                            (1, "metal", "Metal")):
        ax = axes[0, col]
        pids = [pid for pid in paired if by_pdb[pid]["uma_v2"]["kind"] == split_filter]
        if not pids:
            ax.set_title(f"{split_name}  (no paired data yet)")
            continue
        uma_div = [by_pdb[pid]["uma_v2"]["mean_pairwise_hamming_distal"] for pid in pids]
        lig_div = [by_pdb[pid]["ligandmpnn"]["mean_pairwise_hamming_distal"] for pid in pids]
        x_uma = np.full(len(pids), 0)
        x_lig = np.full(len(pids), 1)
        for u, lig in zip(uma_div, lig_div):
            ax.plot([0, 1], [u, lig], color="#888", alpha=0.5, linewidth=0.7)
        ax.scatter(x_uma, uma_div, color="#2C5F8E", s=50, edgecolor="black", linewidth=0.5,
                   label="UMA-v2", zorder=10)
        ax.scatter(x_lig, lig_div, color="#C13C3C", s=50, edgecolor="black", linewidth=0.5,
                   label="LigandMPNN", zorder=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["UMA-v2", "LigandMPNN"], fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("Mean pairwise Hamming distance\n(at distal positions)", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{split_name}  (N = {len(pids)})", fontsize=11)

    # ── Row 2: distal recovery vs distal diversity scatter ─────────────────
    for col, split_filter, split_name in ((0, "small_molecule", "Small molecule"),
                                            (1, "metal", "Metal")):
        ax = axes[1, col]
        pids = [pid for pid in paired if by_pdb[pid]["uma_v2"]["kind"] == split_filter]
        if not pids:
            ax.set_title(f"{split_name}  (no paired data yet)")
            continue
        for pid in pids:
            uma = by_pdb[pid]["uma_v2"]
            lig = by_pdb[pid]["ligandmpnn"]
            ax.plot([uma["mean_distal_recovery"], lig["mean_distal_recovery"]],
                    [uma["mean_pairwise_hamming_distal"], lig["mean_pairwise_hamming_distal"]],
                    color="#888", alpha=0.4, linewidth=0.6)
        for pid in pids:
            uma = by_pdb[pid]["uma_v2"]
            lig = by_pdb[pid]["ligandmpnn"]
            ax.scatter(uma["mean_distal_recovery"], uma["mean_pairwise_hamming_distal"],
                       color="#2C5F8E", s=40, edgecolor="black", linewidth=0.5, zorder=10)
            ax.scatter(lig["mean_distal_recovery"], lig["mean_pairwise_hamming_distal"],
                       color="#C13C3C", s=40, edgecolor="black", linewidth=0.5, zorder=10)
        ax.set_xlabel("Distal recovery (mean over K samples)", fontsize=9)
        ax.set_ylabel("Distal diversity (mean pairwise Hamming)", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{split_name}", fontsize=11)
        ax.legend(["paired (UMA, LigMPNN)", "UMA-v2", "LigandMPNN"], fontsize=8, loc="best")

    fig.suptitle(
        "Pocket-fixed redesign: distal-position diversity and recovery, UMA-v2 vs LigandMPNN",
        fontsize=11, y=1.0,
    )
    plt.tight_layout()
    pdf = args.out_dir / "fig5_pocket_distal.pdf"
    png = args.out_dir / "fig5_pocket_distal.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
