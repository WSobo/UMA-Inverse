"""Figure 7: Ligand-conditioning signal vs. distance to ligand, per class.

Three panels (small molecule, metal, nucleotide). Each plots the mean KL
divergence between the ligand-conditioned and ligand-masked sequence
distributions as a function of residue-to-ligand distance shell, for
UMA-Inverse vs LigandMPNN (±1 SEM). Log-y so the distal collapse is visible.

The headline is distal *persistence*: beyond ~10 Å UMA-Inverse retains a
substantial ligand signal while LigandMPNN's collapses toward zero. (At the
immediate 0-5 Å interface LigandMPNN is comparable or stronger; this is not a
near-pocket-recovery claim.)

Source:
    outputs/preprint/distal_kl/by_class_v5/<class>/distal_kl_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

UMA_COLOR = "#2C5F8E"
LIG_COLOR = "#C13C3C"
BIN_ORDER = ["0-5", "5-10", "10-15", "15-25", ">25"]
XTICKS = [2.5, 7.5, 12.5, 20.0, 30.0]
XLABELS = ["0–5", "5–10", "10–15", "15–25", ">25"]

MODELS = [
    ("uma-inverse-v5", "UMA-Inverse", UMA_COLOR),
    ("ligandmpnn",     "LigandMPNN",  LIG_COLOR),
]
SPLITS = [
    ("small_molecule", "Small molecule"),
    ("metal",          "Metal"),
    ("nucleotide",     "Nucleotide"),
]


def _midpoint(lo: float, hi: float | None) -> float:
    return lo + 5.0 if hi is None else (lo + hi) / 2.0


def _load(path: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            data.setdefault(row["model"], {})[row["distance_bin"]] = row
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--by-class-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "distal_kl" / "by_class_v5",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)

    for ax, (split, title) in zip(axes, SPLITS):
        csv_path = args.by_class_dir / split / "distal_kl_summary.csv"
        if not csv_path.exists():
            ax.set_title(f"{title}\n(missing)", fontsize=10)
            continue
        raw = _load(csv_path)
        n_pdbs = 0
        for model_key, label, color in MODELS:
            if model_key not in raw:
                continue
            bins = raw[model_key]
            xs, ys, errs = [], [], []
            for b in BIN_ORDER:
                if b not in bins:
                    continue
                row = bins[b]
                hi = float(row["hi"]) if row["hi"] else None
                n = float(row["n_residues"])
                xs.append(_midpoint(float(row["lo"]), hi))
                ys.append(float(row["mean_kl"]))
                errs.append(float(row["std_kl"]) / math.sqrt(n) if n > 0 else 0.0)
                if b == "5-10":
                    n_pdbs = max(n_pdbs, int(float(row["n_pdbs"])))
            ax.errorbar(np.array(xs), np.array(ys), yerr=np.array(errs),
                        color=color, linewidth=2, marker="o", markersize=6,
                        capsize=3, label=label, zorder=10)
        ax.set_yscale("log")
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XLABELS, fontsize=9)
        ax.set_xlabel("Distance to nearest ligand atom (Å)", fontsize=10)
        ax.set_title(f"{title}  (N={n_pdbs} PDBs)", fontsize=11)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5, which="both")
        ax.set_axisbelow(True)
        if split == SPLITS[0][0]:
            ax.set_ylabel("Mean KL divergence\n(ligand on vs. masked)", fontsize=10)
            ax.legend(fontsize=10, loc="upper right")

    fig.suptitle(
        "Ligand-conditioning signal vs. distance: UMA-Inverse retains distal signal "
        "where LigandMPNN collapses",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    pdf = args.out_dir / "fig7_distal_signal.pdf"
    png = args.out_dir / "fig7_distal_signal.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
