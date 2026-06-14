"""Figure 6: Boltz-2 cofold ligand-pose RMSD for pocket-fixed redesigns.

Three arms — native crystal sequence (upper bound), UMA-Inverse, LigandMPNN —
cofolded under the identical protocol. We show the best-of-5 ligand-pose RMSD
distribution per arm. Honest result: both design methods are close to the native
floor and confidently folded, but LigandMPNN reproduces the native ligand pose
modestly but significantly better than UMA-Inverse (paired Wilcoxon p=0.003, N=86).

Source:
    outputs/preprint/cofold_metrics_v5_big.csv
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

ARMS = [
    ("native", "Native\nsequence", "#6F8FAF"),
    ("uma_v5", "UMA-Inverse", "#2C5F8E"),
    ("ligandmpnn", "LigandMPNN", "#C13C3C"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics_v5_big.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()
    if not args.metrics.exists():
        raise SystemExit(f"missing {args.metrics} -- run cofold_metrics.py first")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per PDB (median over the design samples) so the plotted
    # distribution matches the per-PDB paired test reported in the text.
    import statistics
    per_pdb: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in csv.DictReader(args.metrics.open()):
        try:
            v = float(r["ligand_rmsd_best"])
        except (KeyError, ValueError):
            continue
        if v == v:
            per_pdb[(r["method"], r["pdb_id"])].append(v)
    vals: dict[str, list[float]] = defaultdict(list)
    for (method, _pdb), vs in per_pdb.items():
        vals[method].append(statistics.median(vs))

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    data, labels, colors, positions = [], [], [], []
    for i, (key, label, color) in enumerate(ARMS):
        if key not in vals:
            continue
        data.append(vals[key])
        labels.append(f"{label}\n(n={len(vals[key])})")
        colors.append(color)
        positions.append(i)

    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)
        pc.set_linewidth(0.7)
    ax.boxplot(data, positions=positions, widths=0.18, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
               medianprops=dict(color="black", linewidth=1.8),
               whiskerprops=dict(color="black", linewidth=0.7),
               capprops=dict(color="black", linewidth=0.7))
    for pos, d in zip(positions, data):
        med = float(np.median(d))
        ax.text(pos + 0.26, med, f"{med:.2f}", va="center", fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Ligand-pose RMSD (Å, best of 5)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Cofold ligand-pose RMSD\n(LigandMPNN vs UMA: paired p=0.003)", fontsize=10)

    plt.tight_layout()
    pdf = args.out_dir / "fig6_cofold.pdf"
    png = args.out_dir / "fig6_cofold.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
