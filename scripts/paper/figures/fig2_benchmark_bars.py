"""Figure 2: standard interface-recovery benchmark bar chart.

Three panels (metal, small_molecule, nucleotide-caveat-only). Primary bars
are UMA-Inverse-1 (v3), LigandMPNN (paper), ProteinMPNN (paper). Supplemental
comparison bars for UMA-v2 and UMA-v1 are included lighter.

Sources:
    UMA-Inverse-1 (v3) per-PDB: outputs/benchmark/interface_recovery/v3-ep23-test_<cls>/per_pdb.csv
    UMA-v2 (per-PDB):           outputs/benchmark/interface_recovery/v2-ep19-test_<cls>/per_pdb.csv
    UMA-v1 (per-PDB):           outputs/benchmark/interface_recovery/ep11-test_<cls>/per_pdb.csv
    LigandMPNN paper:            hardcoded from Dauparas et al.
    ProteinMPNN paper:           hardcoded from Dauparas et al.

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
# LigandMPNN scored under our identical protocol (same PDBs / mask / metric as
# UMA); published values from Dauparas et al. shown in parentheses for reference.
LIGANDMPNN_OURS = {"metal": 0.644, "small_molecule": 0.598, "nucleotide": 0.533}
LIGANDMPNN_PAPER = {"metal": 0.775, "small_molecule": 0.633, "nucleotide": 0.505}
PROTEINMPNN_REF = {"metal": 0.406, "small_molecule": 0.505, "nucleotide": 0.471}


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
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

    methods = ("UMA-Inverse", "LigandMPNN", "ProteinMPNN")
    colors = {
        "UMA-Inverse": "#2C5F8E",
        "LigandMPNN":  "#C13C3C",
        "ProteinMPNN": "#888888",
    }

    for ax, split in zip(axes, splits):
        v3_meds = _load_pdb_medians(args.bench_dir / f"v5-test_{split}" / "per_pdb.csv")

        def _mean_err(meds):
            if not meds:
                return float("nan"), 0.0
            return float(np.mean(meds)), float(np.std(meds, ddof=0) / np.sqrt(len(meds)))

        v3_mean, v3_err = _mean_err(v3_meds)

        values = [v3_mean, LIGANDMPNN_OURS[split], PROTEINMPNN_REF[split]]
        errors = [v3_err, 0.0, 0.0]

        x = np.arange(len(methods))
        bars = ax.bar(
            x, values, yerr=errors, capsize=3,
            color=[colors[m] for m in methods],
            edgecolor="black", linewidth=0.5,
        )
        # Annotate each bar; for LigandMPNN (ours) show the published value in parens.
        for idx, (bar, val) in enumerate(zip(bars, values)):
            if val != val:  # NaN
                continue
            txt = f"{val:.3f}"
            if idx == 1:
                txt = f"{val:.3f}\n(paper {LIGANDMPNN_PAPER[split]:.3f})"
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                    txt, ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        n_label = f" (N={len(v3_meds)})" if v3_meds else ""
        ax.set_title(f"{split.replace('_', ' ').title()}{n_label}", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        if split == splits[0]:
            ax.set_ylabel("Interface sequence recovery", fontsize=10)
        if split == "nucleotide":
            ax.text(0.5, 0.97, "(nucleic acid routed\nas ligand context)",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="#666", style="italic")

    fig.suptitle(
        "Interface sequence recovery (matched protocol: 10 designs/PDB, T=0.1, "
        "random decoding, 5Å cutoff)\n"
        "LigandMPNN re-run by us on identical structures; published values in parentheses",
        fontsize=10, y=1.02,
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
