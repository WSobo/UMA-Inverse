"""Figure 6: Boltz-2 cofold metrics for pocket-fixed redesigns.

Four-panel figure (2x2):
  Top-left:    ipTM distribution per method per split (violin)
  Top-right:   Pocket-only Calpha RMSD distribution per method per split (violin)
  Bottom-left: Ligand-pose RMSD distribution per method per split (violin)
  Bottom-right: Predicted-affinity scatter, UMA vs LigandMPNN per PDB (paired)

Caption explicitly notes: "Boltz-2 ipTM, plDDT, and predicted affinity are
model predictions, not measurements. Used here for relative comparison
between two design methods on the same scaffold."

Source:
    outputs/preprint/cofold_metrics.csv
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


def _violin_panel(ax, rows, field, ylabel, title):
    """Side-by-side violins: small_molecule (UMA, LigMPNN) | metal (UMA, LigMPNN)."""
    data: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        data[(r["kind"], r["method"])].append(v)

    positions, labels, vals, colors = [], [], [], []
    for x, kind in enumerate(("small_molecule", "metal")):
        for offset, method in enumerate(("uma_v2", "ligandmpnn")):
            positions.append(x * 3 + offset)
            labels.append(f"{method.replace('_', '-')}\n({kind})")
            vals.append(data[(kind, method)] or [0])
            colors.append("#2C5F8E" if method == "uma_v2" else "#C13C3C")
    if not any(v != [0] for v in vals):
        ax.set_title(f"{title}\n(no cofold data yet)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        return
    parts = ax.violinplot(vals, positions=positions, widths=0.7,
                           showmeans=False, showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.65)
        pc.set_linewidth(0.7)
    ax.boxplot(vals, positions=positions, widths=0.18, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
               medianprops=dict(color="black", linewidth=1.5),
               whiskerprops=dict(color="black", linewidth=0.7),
               capprops=dict(color="black", linewidth=0.7))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        raise SystemExit(
            f"missing {args.metrics}. Run scripts/preprint/cofold_metrics.py "
            "after Phase B's Boltz-2 cofolds finish."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load(args.metrics)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    _violin_panel(axes[0, 0], rows, "iptm_best",
                   "ipTM (best of 5 diffusion samples)",
                   "Cofold confidence (Boltz-2 ipTM)")
    _violin_panel(axes[0, 1], rows, "pocket_calpha_rmsd_best",
                   "Pocket Cα RMSD (Å, best of 5)",
                   "Pocket geometry preservation")
    _violin_panel(axes[1, 0], rows, "ligand_rmsd_best",
                   "Ligand-pose RMSD (Å, best of 5)",
                   "Ligand-pose preservation")

    # Bottom-right: predicted-affinity paired scatter
    ax = axes[1, 1]
    by_pdb: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get("affinity_pred_value")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        by_pdb[r["pdb_id"]][r["method"]].append(v)
    pdbs_paired = [pid for pid, d in by_pdb.items()
                   if "uma_v2" in d and "ligandmpnn" in d]
    if not pdbs_paired:
        ax.set_title("Predicted affinity (UMA vs LigandMPNN, paired by PDB)\n(no paired data yet)",
                     fontsize=10)
    else:
        uma_means = [np.mean(by_pdb[pid]["uma_v2"]) for pid in pdbs_paired]
        lig_means = [np.mean(by_pdb[pid]["ligandmpnn"]) for pid in pdbs_paired]
        ax.scatter(lig_means, uma_means, s=60, color="#444", edgecolor="black",
                   linewidth=0.7, alpha=0.85)
        # 1:1 line
        lim = [min(min(uma_means), min(lig_means)) - 0.2,
               max(max(uma_means), max(lig_means)) + 0.2]
        ax.plot(lim, lim, color="#888", linestyle="--", linewidth=0.7, alpha=0.7,
                label="1:1")
        ax.set_xlabel("LigandMPNN designs: predicted affinity\n(Boltz-2; mean over 5 samples)",
                       fontsize=9)
        ax.set_ylabel("UMA-v2 designs: predicted affinity\n(Boltz-2; mean over 5 samples)",
                       fontsize=9)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title("Predicted affinity (paired by PDB)", fontsize=10)

    fig.suptitle(
        "Boltz-2 cofold of pocket-fixed redesigns -- predicted, not measured "
        "(used for relative comparison only)",
        fontsize=11, y=1.0,
    )
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
