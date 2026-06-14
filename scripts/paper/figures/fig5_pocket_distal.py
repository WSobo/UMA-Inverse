"""Figure 5: pocket-fixed distal sequence recovery, UMA-Inverse vs LigandMPNN.

With the native pocket held fixed, both methods redesign the remaining (distal)
positions. We plot per-PDB distal recovery (mean over samples) by ligand class.
Honest result: LigandMPNN recovers more of the distal positions than UMA-Inverse.
Diversity is reported in the text with a temperature caveat (not shown here, since
a single-temperature cross-model diversity comparison is not controlled).

Source:
    outputs/preprint/pocket_fixed_v5/pocket_fixed_summary.csv
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

UMA_COLOR = "#2C5F8E"
LIG_COLOR = "#C13C3C"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pocket_fixed_v5" / "pocket_fixed_summary.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()
    if not args.summary.exists():
        raise SystemExit(f"missing {args.summary} -- run compute_pocket_fixed_metrics.py first")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in csv.DictReader(args.summary.open()):
        try:
            v = float(r["mean_distal_recovery"])
        except (KeyError, ValueError):
            continue
        if v == v:
            by[(r["kind"], r["method"])].append(v)

    kinds = ["small_molecule", "metal"]
    methods = [("uma_v5", "UMA-Inverse", UMA_COLOR), ("ligandmpnn", "LigandMPNN", LIG_COLOR)]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = np.arange(len(kinds))
    w = 0.38
    for i, (key, label, color) in enumerate(methods):
        means, errs = [], []
        for kind in kinds:
            vals = np.array(by.get((kind, key), [np.nan]))
            means.append(float(np.nanmean(vals)))
            n = int(np.sum(~np.isnan(vals)))
            errs.append(float(np.nanstd(vals) / np.sqrt(n)) if n > 0 else 0.0)
        bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=3,
                      color=color, edgecolor="black", linewidth=0.5, label=label)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.012, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Small molecule", "Metal"], fontsize=10)
    ax.set_ylabel("Pocket-fixed distal recovery\n(per-PDB mean)", fontsize=10)
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Pocket-fixed distal recovery (N=104)", fontsize=11)

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
