"""Figure 5: pocket-fixed redesign — distal recovery and diversity comparison.

2×2 layout:
  Rows:    distal recovery (top) | distal diversity (bottom)
  Columns: small-molecule (left) | metal (right)

Each point is one PDB; connecting lines show within-PDB shifts.

Source:
    outputs/preprint/pocket_fixed_summary.csv
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
            f"missing {args.summary}. Run scripts/paper/compute_pocket_fixed_metrics.py first."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load(args.summary)
    by_pdb: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_pdb[r["pdb_id"]][r["method"]] = r
    paired = [pid for pid, d in by_pdb.items() if "uma_v3" in d and "ligandmpnn" in d]

    metrics = [
        ("mean_distal_recovery",        "Distal recovery (mean over 20 samples)"),
        ("mean_pairwise_hamming_distal", "Distal diversity (mean pairwise Hamming)"),
    ]
    kinds = [
        ("small_molecule", "Small molecule"),
        ("metal",          "Metal"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, (kind, kind_label) in enumerate(kinds):
            ax = axes[row_idx, col_idx]
            pids_k = [pid for pid in paired if by_pdb[pid]["uma_v3"]["kind"] == kind]

            uma_vals = [by_pdb[pid]["uma_v3"][metric] for pid in pids_k]
            lig_vals = [by_pdb[pid]["ligandmpnn"][metric] for pid in pids_k]

            all_vals = uma_vals + lig_vals
            if all_vals:
                pad = (max(all_vals) - min(all_vals)) * 0.10 or 0.02
                ax.set_ylim(max(0.0, min(all_vals) - pad), min(1.0, max(all_vals) + pad))

            # Connecting lines behind points
            for u, lig in zip(uma_vals, lig_vals):
                ax.plot([0, 1], [u, lig], color="#555", alpha=0.20, linewidth=0.8)

            ax.scatter(np.zeros(len(pids_k)), uma_vals,
                       color=UMA_COLOR, s=40, edgecolor="white", linewidth=0.4,
                       zorder=10, label=f"UMA-Inverse-1 (N={len(pids_k)})")
            ax.scatter(np.ones(len(pids_k)), lig_vals,
                       color=LIG_COLOR, s=40, edgecolor="white", linewidth=0.4,
                       zorder=10, label=f"LigandMPNN (N={len(pids_k)})")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["UMA-Inverse-1", "LigandMPNN"], fontsize=10)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{kind_label}", fontsize=11)
            ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Pocket-fixed redesign: UMA-Inverse-1 vs LigandMPNN",
        fontsize=12, y=1.01,
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
