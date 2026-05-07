"""Figure 7: distal-confidence vs cofold-quality correlation, UMA vs LigandMPNN.

Operationalises Test 4 from `distal_signal_analysis.py` as a figure.

Per-PDB distal confidence (1 - mean_pairwise_hamming_distal) on x;
Boltz-2 cofold quality on y. Each row is a different quality metric;
columns are method (UMA-v2 left, LigandMPNN right).

The asymmetry visible across both rows is the §3.4 finding: UMA's distal
confidence tracks pocket-specific cofold confidence (ipTM); LigandMPNN's
distal confidence is essentially decoupled from ipTM, while still picking
up the global-fold signal (pLDDT) that any sequence-design recovery
correlates with.

Source:
    outputs/preprint/pocket_fixed_summary.csv
    outputs/preprint/cofold_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

UMA_COLOR = "#2C5F8E"
LIG_COLOR = "#C13C3C"

# (csv_field, axis_label)
QUALITY_METRICS = [
    ("iptm_best",          "Boltz-2 ipTM (best of 5)"),
    ("complex_plddt_best", "Boltz-2 complex pLDDT (best of 5)"),
]


def _agg(rows: list[dict], field: str) -> float:
    vals = []
    for r in rows:
        v = r.get(field, "")
        if v in ("", "nan"):
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if f != f:  # NaN
            continue
        vals.append(f)
    return statistics.fmean(vals) if vals else float("nan")


def _pick_corner(cs: np.ndarray, qs: np.ndarray) -> tuple[float, float, str, str]:
    """Return (x_axes, y_axes, ha, va) for the panel corner with the fewest
    data points nearby, so the stats annotation never overlaps the scatter."""
    candidates = [
        (0.04, 0.96, "left",  "top"),     # upper-left
        (0.96, 0.96, "right", "top"),     # upper-right
        (0.04, 0.04, "left",  "bottom"),  # lower-left
        (0.96, 0.04, "right", "bottom"),  # lower-right
    ]
    x_lo, x_hi = cs.min(), cs.max()
    y_lo, y_hi = qs.min(), qs.max()
    x_span = max(x_hi - x_lo, 1e-9)
    y_span = max(y_hi - y_lo, 1e-9)
    cs_norm = (cs - x_lo) / x_span
    qs_norm = (qs - y_lo) / y_span
    box_half = 0.18  # ±18% axes-space window around each corner
    best = candidates[0]
    best_count = len(cs) + 1
    for ax_x, ax_y, ha, va in candidates:
        in_box = ((cs_norm >= ax_x - box_half) & (cs_norm <= ax_x + box_half)
                  & (qs_norm >= ax_y - box_half) & (qs_norm <= ax_y + box_half))
        count = int(in_box.sum())
        if count < best_count:
            best_count = count
            best = (ax_x, ax_y, ha, va)
    return best


def _scatter_panel(ax, conf, quality, *, color, method_label, ylabel, show_xlabel):
    """One scatter panel with linear regression + r/p annotation."""
    valid = [(c, q) for c, q in zip(conf, quality) if not (c != c or q != q)]
    if len(valid) < 5:
        ax.set_title(f"{method_label}  (insufficient data)", fontsize=10)
        return
    cs, qs = zip(*valid)
    cs = np.asarray(cs)
    qs = np.asarray(qs)

    ax.scatter(cs, qs, s=55, color=color, edgecolor="black", linewidth=0.5,
               alpha=0.85, zorder=10)

    # Linear regression line
    slope, intercept = np.polyfit(cs, qs, 1)
    xline = np.linspace(cs.min(), cs.max(), 50)
    yline = slope * xline + intercept
    ax.plot(xline, yline, color=color, linewidth=1.6, alpha=0.85, zorder=8)

    # Stats annotation (Pearson + Spearman) — placed in the emptiest corner
    pr, pp = pearsonr(cs, qs)
    sr, _sp = spearmanr(cs, qs)
    sig = ""
    if pp < 0.001:
        sig = " ***"
    elif pp < 0.01:
        sig = " **"
    elif pp < 0.05:
        sig = " *"
    text = (f"Pearson  r = {pr:+.2f}{sig}\n"
            f"p = {pp:.3g}\n"
            f"Spearman ρ = {sr:+.2f}\n"
            f"N = {len(valid)}")
    x_anchor, y_anchor, ha, va = _pick_corner(cs, qs)
    ax.text(x_anchor, y_anchor, text, transform=ax.transAxes, fontsize=8.5,
            horizontalalignment=ha, verticalalignment=va,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

    if show_xlabel:
        ax.set_xlabel("Per-PDB distal confidence  (1 − mean pairwise Hamming, distal positions)",
                       fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(method_label, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pocket_fixed_summary.csv",
    )
    parser.add_argument(
        "--cofold",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()

    for path in (args.summary, args.cofold):
        if not path.exists():
            raise SystemExit(
                f"missing {path}. Generate it first via the scripts/paper pipeline."
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load summary (per-PDB-per-method confidence) ────────────────────────
    pf_by: dict[str, dict[str, dict]] = defaultdict(dict)
    with args.summary.open() as f:
        for row in csv.DictReader(f):
            pf_by[row["pdb_id"]][row["method"]] = row

    # ── Load cofold metrics (multiple samples per PDB, aggregate to mean) ───
    cf_by: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with args.cofold.open() as f:
        for row in csv.DictReader(f):
            cf_by[row["pdb_id"]][row["method"]].append(row)

    # ── Restrict to small-molecule PDBs paired across both phases ──────────
    pdbs = sorted(
        p for p, d in pf_by.items()
        if d.get("uma_v2", {}).get("kind") == "small_molecule"
        and "ligandmpnn" in d
        and "uma_v2" in cf_by[p]
        and "ligandmpnn" in cf_by[p]
    )
    if not pdbs:
        raise SystemExit("no PDBs with both pocket-fixed and cofold data — nothing to plot")

    # ── Build per-method (conf, quality) vectors ───────────────────────────
    conf_by_method: dict[str, list[float]] = {
        m: [1.0 - float(pf_by[p][m]["mean_pairwise_hamming_distal"]) for p in pdbs]
        for m in ("uma_v2", "ligandmpnn")
    }

    # ── Plot grid: rows = metric, cols = method ────────────────────────────
    n_rows = len(QUALITY_METRICS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.4 * n_rows), squeeze=False)

    for row_idx, (field, ylabel) in enumerate(QUALITY_METRICS):
        for col_idx, (method, color, mlabel) in enumerate((
            ("uma_v2",     UMA_COLOR, "UMA-v2"),
            ("ligandmpnn", LIG_COLOR, "LigandMPNN"),
        )):
            ax = axes[row_idx, col_idx]
            quality = [_agg(cf_by[p][method], field) for p in pdbs]
            _scatter_panel(
                ax,
                conf_by_method[method],
                quality,
                color=color,
                method_label=mlabel,
                ylabel=ylabel,
                show_xlabel=(row_idx == n_rows - 1),
            )

    fig.suptitle(
        "Distal-confidence vs Boltz-2 cofold quality — small-molecule split, "
        "pocket-fixed redesign",
        fontsize=11, y=1.0,
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
