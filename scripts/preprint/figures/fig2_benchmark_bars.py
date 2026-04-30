"""Figure 2: standard interface-recovery benchmark bar chart.

Three panels (metal, small_molecule, nucleotide-caveat-only). Four bars per
panel: UMA-v2, UMA-v1, LigandMPNN (paper), ProteinMPNN (paper).

Sources:
    UMA-v2 (per-PDB):  outputs/benchmark/interface_recovery/v2-ep19-test_<cls>/per_pdb.csv
    UMA-v1 (per-PDB):  outputs/benchmark/interface_recovery/ep11-test_<cls>/per_pdb.csv
    LigandMPNN paper:  hardcoded from Dauparas et al.
    ProteinMPNN paper: hardcoded from Dauparas et al.

Output:
    outputs/preprint/figures/fig2_benchmark_bars.{pdf,png}
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

# Paper numbers from Dauparas et al. (LigandMPNN paper) — they report MEDIAN
# of per-PDB recoveries; we report MEAN-of-per-PDB-medians for UMA. The chart
# caption notes the convention difference.
LIGANDMPNN_REF = {"metal": 0.775, "small_molecule": 0.633, "nucleotide": 0.505}
PROTEINMPNN_REF = {"metal": 0.406, "small_molecule": 0.505, "nucleotide": 0.340}


def _load_pdb_medians(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        return []
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["median_recovery"])
                if v == v:  # skip NaN
                    out.append(v)
            except (KeyError, ValueError):
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark" / "interface_recovery",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = ("metal", "small_molecule", "nucleotide")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)

    methods = ("UMA-v2", "UMA-v1", "LigandMPNN", "ProteinMPNN")
    colors = {
        "UMA-v2":      "#2C5F8E",
        "UMA-v1":      "#7CA9CD",
        "LigandMPNN":  "#C13C3C",
        "ProteinMPNN": "#888888",
    }

    for ax, split in zip(axes, splits):
        v2_meds = _load_pdb_medians(args.bench_dir / f"v2-ep19-test_{split}" / "per_pdb.csv")
        v1_meds = _load_pdb_medians(args.bench_dir / f"ep11-test_{split}" / "per_pdb.csv")

        v2_mean = float(np.mean(v2_meds)) if v2_meds else float("nan")
        v2_err = float(np.std(v2_meds, ddof=0) / np.sqrt(len(v2_meds))) if v2_meds else 0.0
        v1_mean = float(np.mean(v1_meds)) if v1_meds else float("nan")
        v1_err = float(np.std(v1_meds, ddof=0) / np.sqrt(len(v1_meds))) if v1_meds else 0.0

        values = [v2_mean, v1_mean, LIGANDMPNN_REF[split], PROTEINMPNN_REF[split]]
        errors = [v2_err, v1_err, 0.0, 0.0]  # paper numbers: no per-PDB error available

        x = np.arange(len(methods))
        bars = ax.bar(
            x, values, yerr=errors, capsize=3,
            color=[colors[m] for m in methods],
            edgecolor="black", linewidth=0.5,
        )
        # Annotate each bar with its value
        for bar, val in zip(bars, values):
            if val == val:  # not NaN
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        n_label = f" (N={len(v2_meds)})" if v2_meds else ""
        ax.set_title(f"{split.replace('_', ' ').title()}{n_label}", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        if split == splits[0]:
            ax.set_ylabel("Interface sequence recovery", fontsize=10)
        # Highlight if this is the nucleotide caveat panel
        if split == "nucleotide":
            ax.text(0.5, 0.97, "(featurizer ablation;\nnot a fair comparison)",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="#666", style="italic")

    fig.suptitle(
        "Interface sequence recovery, LigandMPNN-paper protocol\n"
        "(10 designs/PDB, T=0.1, random decoding, 5Å sidechain cutoff)",
        fontsize=11, y=1.0,
    )
    plt.tight_layout()

    pdf = args.out_dir / "fig2_benchmark_bars.pdf"
    png = args.out_dir / "fig2_benchmark_bars.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
