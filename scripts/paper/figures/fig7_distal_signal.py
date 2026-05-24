"""Figure 7 (PDF Figure 5): Ligand-distal KL signal by distance shell.

Two-line plot of mean KL divergence (ligand-conditioned vs. ligand-ablated)
as a function of residue distance to the nearest ligand atom.  UMA-Inverse-1
vs LigandMPNN, ±1 SEM error bars.

Source:
    outputs/preprint/distal_kl/mechanism/distal_kl_summary.csv
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


def _midpoint(lo: float, hi: float | None) -> float:
    """Return shell midpoint; for open-ended last bin use lo+5."""
    if hi is None:
        return lo + 5.0
    return (lo + hi) / 2.0


def _load(path: Path) -> dict[str, dict]:
    """Return {model: {distance_bin: row_dict}}."""
    data: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            m = row["model"]
            if m not in data:
                data[m] = {}
            data[m][row["distance_bin"]] = row
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kl-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "distal_kl"
                / "mechanism" / "distal_kl_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()

    if not args.kl_csv.exists():
        raise SystemExit(f"missing {args.kl_csv} — run scripts/paper/distal_kl_shift.py first")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load(args.kl_csv)

    # Ordered bins (match CSV row order).
    bin_order = ["0-5", "5-10", "10-15", "15-25", ">25"]

    models = [
        ("uma-inverse-v3", "UMA-Inverse-1", UMA_COLOR),
        ("ligandmpnn",     "LigandMPNN",    LIG_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model_key, label, color in models:
        if model_key not in raw:
            continue
        bins = raw[model_key]
        xs, ys, errs = [], [], []
        for b in bin_order:
            if b not in bins:
                continue
            row = bins[b]
            lo = float(row["lo"])
            hi = float(row["hi"]) if row["hi"] else None
            n  = float(row["n_residues"])
            mean_kl = float(row["mean_kl"])
            std_kl  = float(row["std_kl"])
            xs.append(_midpoint(lo, hi))
            ys.append(mean_kl)
            errs.append(std_kl / math.sqrt(n) if n > 0 else 0.0)

        xs_arr  = np.array(xs)
        ys_arr  = np.array(ys)
        err_arr = np.array(errs)

        ax.plot(xs_arr, ys_arr, color=color, linewidth=2, marker="o",
                markersize=6, label=label, zorder=10)
        ax.fill_between(xs_arr, ys_arr - err_arr, ys_arr + err_arr,
                        color=color, alpha=0.15, zorder=5)

    ax.set_ylim(bottom=0)

    ax.set_xlabel("Distance to nearest ligand atom (Å)", fontsize=10)
    ax.set_ylabel("Mean KL divergence\n(ligand-conditioned vs. ablated)", fontsize=10)
    ax.set_title("Validation-split sample (43 PDBs)  -  teacher-forced ligand ablation",
                 fontsize=9, color="0.35")
    ax.set_xticks([2.5, 7.5, 12.5, 20.0, 30.0])
    ax.set_xticklabels(["0–5", "5–10", "10–15", "15–25", ">25"], fontsize=9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

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
