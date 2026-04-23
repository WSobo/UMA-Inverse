"""Violin plots of UMA-Inverse interface sequence recovery.

Mirrors the paper-style figure in LigandMPNN: one violin per ligand class
showing the distribution of per-PDB median recoveries, with individual points
overlaid and horizontal reference lines for the LigandMPNN and ProteinMPNN
published numbers.

Also produces a per-sample variant using ``per_sample.csv`` — the per-PDB
medians violin is the fair comparison to the paper, the per-sample violin is
useful internally for understanding sampling variance.

Outputs:
    outputs/benchmark/interface_recovery/violins_per_pdb.png / .pdf
    outputs/benchmark/interface_recovery/violins_per_sample.png / .pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "benchmark" / "interface_recovery"

# Only the two classes the current model is trained to support (see
# summarize_test_benchmarks.py for the nucleotide caveat).
COMPARED_RUNS = [
    ("metal",           "ep11-test_metal"),
    ("small_molecule",  "ep11-test_small_molecule"),
]

LIGANDMPNN_REF = {"metal": 0.775, "small_molecule": 0.633}
PROTEINMPNN_REF = {"metal": 0.406, "small_molecule": 0.505}

# Class-name → pretty label for the axis tick.
PRETTY = {"metal": "Metals", "small_molecule": "Small molecules"}


def _read_column(csv_path: Path, column: str) -> list[float]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            v = row.get(column, "")
            if v == "" or v == "nan":
                continue
            x = float(v)
            if x != x:  # NaN
                continue
            out.append(x)
    return out


def _violin(ax, data_by_class: dict[str, list[float]],
            title: str, point_label: str) -> None:
    """Shared violin-drawing routine."""
    classes = [c for c in data_by_class if data_by_class[c]]
    data = [data_by_class[c] for c in classes]
    positions = list(range(1, len(classes) + 1))

    # Main violins (filled, wider)
    vp = ax.violinplot(data, positions=positions, showmeans=False,
                       showmedians=True, showextrema=False, widths=0.75)
    for body in vp["bodies"]:
        body.set_facecolor("#4C72B0")
        body.set_edgecolor("#1F3C5A")
        body.set_alpha(0.55)
    if "cmedians" in vp:
        vp["cmedians"].set_color("#1F3C5A")
        vp["cmedians"].set_linewidth(1.5)

    # Individual points with horizontal jitter
    rng = np.random.default_rng(0)
    for i, (pos, xs) in enumerate(zip(positions, data)):
        jitter = rng.normal(0.0, 0.04, size=len(xs))
        ax.scatter(np.full(len(xs), pos) + jitter, xs,
                   s=8, alpha=0.35, color="#1F3C5A", zorder=3,
                   label=point_label if i == 0 else None)

    # Reference lines
    for i, (pos, cls) in enumerate(zip(positions, classes)):
        ax.hlines(LIGANDMPNN_REF[cls], pos - 0.35, pos + 0.35,
                  colors="#C0392B", linestyles="-", linewidth=2.0,
                  label="LigandMPNN" if i == 0 else None, zorder=4)
        ax.hlines(PROTEINMPNN_REF[cls], pos - 0.35, pos + 0.35,
                  colors="#888888", linestyles="--", linewidth=1.5,
                  label="ProteinMPNN" if i == 0 else None, zorder=4)

    # UMA mean marker
    for i, (pos, xs) in enumerate(zip(positions, data)):
        mean = float(np.mean(xs))
        ax.scatter([pos], [mean], marker="D", s=60, color="#E8A33D",
                   edgecolor="#7A4B00", linewidth=1.0,
                   label="UMA mean" if i == 0 else None, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([PRETTY.get(c, c) for c in classes])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Sidechain-interface sequence recovery")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # Per-PDB medians (one dot per PDB) — this is the direct paper analog
    per_pdb: dict[str, list[float]] = {}
    per_sample: dict[str, list[float]] = {}
    missing: list[str] = []

    for cls, run in COMPARED_RUNS:
        run_dir = args.out_dir / run
        per_pdb_csv = run_dir / "per_pdb.csv"
        per_sample_csv = run_dir / "per_sample.csv"
        if not per_pdb_csv.exists():
            missing.append(str(per_pdb_csv))
            continue
        per_pdb[cls] = _read_column(per_pdb_csv, "median_recovery")
        if per_sample_csv.exists():
            per_sample[cls] = _read_column(per_sample_csv, "interface_recovery")

    if missing:
        print("!! missing inputs:")
        for m in missing:
            print(f"   - {m}")
        if not per_pdb:
            raise SystemExit(1)

    # ── Figure 1: per-PDB medians (one violin per class, one dot per PDB) ─
    fig, ax = plt.subplots(figsize=(7, 5))
    _violin(ax, per_pdb,
            title="UMA-Inverse (stage-3 ep11): interface recovery by ligand class\n"
                  "violin = per-PDB median across 10 AR samples at T=0.1",
            point_label="per-PDB median")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = args.out_dir / f"violins_per_pdb.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)

    # ── Figure 2: per-sample raw recoveries (10 dots per PDB) ─────────────
    if per_sample:
        fig, ax = plt.subplots(figsize=(7, 5))
        _violin(ax, per_sample,
                title="UMA-Inverse (stage-3 ep11): per-sample interface recovery\n"
                      "violin = all 10 AR samples × every PDB (T=0.1)",
                point_label="per-sample")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out = args.out_dir / f"violins_per_sample.{ext}"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"wrote {out}")
        plt.close(fig)

    # Print a one-line summary for sanity checking
    print()
    for cls in per_pdb:
        n = len(per_pdb[cls])
        mean = float(np.mean(per_pdb[cls]))
        print(f"{cls:16s} N={n:>3d}  UMA mean = {mean:.3f}   "
              f"LigandMPNN = {LIGANDMPNN_REF[cls]:.3f}   "
              f"ProteinMPNN = {PROTEINMPNN_REF[cls]:.3f}")


if __name__ == "__main__":
    main()
