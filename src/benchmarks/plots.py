"""Matplotlib figure generation for the paper.

Six publication-grade PNGs are written to ``<out_dir>/figures/``:

* ``confusion.png`` — 20×20 confusion matrix heatmap (normalised by row).
* ``calibration.png`` — reliability diagram with the diagonal overlaid.
* ``near_ligand.png`` — recovery as a function of distance to the
  nearest ligand atom (the main selling point of a ligand-conditioned
  model).
* ``temperature_diversity.png`` — dual-axis plot of recovery vs
  Hamming diversity across sampling temperatures.
* ``aa_composition.png`` — native vs predicted per-AA frequency.
* ``perplexity_by_length.png`` — per-PDB perplexity scatter against
  residue count (catches length-related pathologies).

All figures use a consistent style block at the top so regenerating from
different CSVs produces visually matching output.

Each function takes already-aggregated data (not raw per-position
records) so the plotting code is trivially testable with synthetic
inputs.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.metrics import CalibrationBin

logger = logging.getLogger(__name__)

# ─── Global style ─────────────────────────────────────────────────────────────

matplotlib.use("Agg")  # headless — no display needed for cluster runs
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "figure.figsize": (6.4, 4.8),
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


# ─── 1. Confusion matrix ──────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm_normalised: np.ndarray,
    aa_labels: Sequence[str],
    path: Path,
    *,
    title: str = "Confusion matrix (row-normalised)",
) -> None:
    """Heatmap of row-normalised confusion counts.

    Row = native AA, column = predicted AA. Diagonal dominance indicates
    accurate recovery; off-diagonal clusters indicate systematic
    confusions (e.g. L↔V is common for hydrophobic substitution).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_normalised, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(aa_labels)))
    ax.set_yticks(range(len(aa_labels)))
    ax.set_xticklabels(aa_labels, fontsize=9)
    ax.set_yticklabels(aa_labels, fontsize=9)
    ax.set_xlabel("Predicted AA")
    ax.set_ylabel("Native AA")
    ax.set_title(title)

    # Annotate diagonal (recovery rate per AA) so the figure is readable
    # without needing the raw CSV alongside.
    for i in range(cm_normalised.shape[0]):
        val = cm_normalised[i, i]
        colour = "white" if val < 0.5 else "black"
        ax.text(i, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=colour)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction")
    _save(fig, path)


# ─── 2. Calibration / reliability diagram ────────────────────────────────────


def plot_calibration(
    bins: Sequence[CalibrationBin],
    path: Path,
    *,
    ece: float | None = None,
    title: str = "Confidence calibration",
) -> None:
    """Reliability diagram: predicted probability vs observed accuracy.

    The dashed diagonal is perfect calibration; bars above the diagonal
    are under-confident, below the diagonal are over-confident.
    """
    centres = [(b.lower + b.upper) / 2 for b in bins]
    accs = [b.mean_accuracy for b in bins]
    counts = [b.count for b in bins]
    predicted = [b.mean_predicted_prob for b in bins]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.4, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    bar_colours = ["steelblue" if p >= a else "tomato" for p, a in zip(predicted, accs)]
    ax_top.bar(centres, accs, width=(1 / len(bins)) * 0.9, alpha=0.8, color=bar_colours,
               edgecolor="black", linewidth=0.5, label="Observed accuracy")
    ax_top.scatter(centres, predicted, marker="o", color="black", zorder=5,
                   s=14, label="Mean predicted prob")
    ax_top.set_ylabel("Accuracy / mean predicted prob")
    ax_top.set_ylim(0, 1)
    ax_top.set_title(
        title + (f"   (ECE = {ece:.4f})" if ece is not None else "")
    )
    ax_top.legend(loc="upper left")

    ax_bot.bar(centres, counts, width=(1 / len(bins)) * 0.9,
               color="grey", alpha=0.6, edgecolor="black", linewidth=0.5)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("Predicted probability bin")
    ax_bot.set_ylabel("Count (log)")

    _save(fig, path)


# ─── 3. Recovery vs distance-to-ligand ────────────────────────────────────────


def plot_near_ligand_recovery(
    per_position_df: pd.DataFrame,
    path: Path,
    *,
    bin_edges: Sequence[float] = (0, 4, 6, 8, 10, 15, 20, 30, float("inf")),
    title: str = "Sequence recovery by distance to nearest ligand atom",
) -> None:
    """Recovery broken down by distance buckets.

    Expects columns: ``distance_to_ligand``, ``native_token``, ``pred_token``,
    ``ligand_context_masked`` (if present, both panels are drawn).
    """
    df = per_position_df.copy()
    df["correct"] = df["native_token"] == df["pred_token"]
    df["distance_bin"] = pd.cut(
        df["distance_to_ligand"], bins=list(bin_edges), right=False, include_lowest=True
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Separate curves for ligand-aware vs masked if ablation data present
    has_ablation = (
        "ligand_context_masked" in df.columns
        and df["ligand_context_masked"].nunique() > 1
    )

    if has_ablation:
        masked_col = df["ligand_context_masked"]
        subsets = [
            ("with ligand", df[~masked_col], "steelblue", "o"),
            ("masked", df[masked_col], "tomato", "s"),
        ]
    else:
        subsets = [("", df, "steelblue", "o")]

    for label, sub, colour, marker in subsets:
        grouped = sub.groupby("distance_bin", observed=True)["correct"].agg(["mean", "count"])
        centres = [
            (iv.left + iv.right) / 2 if np.isfinite(iv.right) else iv.left + 2
            for iv in grouped.index
        ]
        ax.plot(
            centres, grouped["mean"], marker=marker, linewidth=1.6,
            color=colour, label=label or "recovery",
        )
        for x, y, n in zip(centres, grouped["mean"], grouped["count"]):
            ax.annotate(
                f"n={int(n)}",
                xy=(x, y),
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                color=colour,
            )

    ax.set_xlabel("Distance to nearest ligand atom (Å)")
    ax.set_ylabel("Sequence recovery")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    if has_ablation:
        ax.legend()
    _save(fig, path)


# ─── 4. Temperature / diversity curve ─────────────────────────────────────────


def plot_temperature_diversity(temperature_df: pd.DataFrame, path: Path) -> None:
    """Dual-axis plot: recovery vs Hamming diversity across T.

    Classic diversity-quality trade-off. Best operating point is usually
    where the two curves cross (enough diversity without collapsing
    recovery).
    """
    df = temperature_df.sort_values("temperature")
    fig, ax_left = plt.subplots(figsize=(6.8, 4.5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        df["temperature"], df["mean_recovery"],
        "o-", color="steelblue", linewidth=1.8, label="Recovery",
    )
    ax_left.fill_between(
        df["temperature"],
        df["mean_recovery"] - df["std_recovery"],
        df["mean_recovery"] + df["std_recovery"],
        alpha=0.2, color="steelblue",
    )
    ax_left.set_ylabel("Sequence recovery", color="steelblue")
    ax_left.tick_params(axis="y", labelcolor="steelblue")
    ax_left.set_ylim(0, 1)
    ax_left.spines["right"].set_visible(True)

    ax_right.plot(
        df["temperature"], df["mean_hamming_diversity"],
        "s--", color="tomato", linewidth=1.6, label="Diversity",
    )
    ax_right.set_ylabel("Mean Hamming diversity", color="tomato")
    ax_right.tick_params(axis="y", labelcolor="tomato")
    ax_right.set_ylim(0, max(1.0, df["mean_hamming_diversity"].max() * 1.1))

    ax_left.set_xlabel("Temperature")
    ax_left.set_title("Sampling temperature: quality vs diversity")
    _save(fig, path)


# ─── 5. AA composition ────────────────────────────────────────────────────────


def plot_aa_composition(
    native_freqs: dict[str, float],
    predicted_freqs: dict[str, float],
    path: Path,
) -> None:
    """Grouped bars comparing native vs predicted AA frequencies."""
    aas = sorted(native_freqs.keys())
    native = [native_freqs[a] for a in aas]
    predicted = [predicted_freqs.get(a, 0.0) for a in aas]

    x = np.arange(len(aas))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, native, width, label="Native", color="steelblue",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, predicted, width, label="Predicted (argmax)", color="tomato",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(aas)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Amino acid")
    ax.set_title("AA composition: native vs predicted")
    ax.legend()
    _save(fig, path)


# ─── 6. Perplexity vs chain length ────────────────────────────────────────────


def plot_perplexity_by_length(per_pdb_df: pd.DataFrame, path: Path) -> None:
    """Per-PDB perplexity scatter against residue count.

    Expects columns: ``num_residues``, ``mean_log_prob``. Computes
    perplexity as ``exp(-mean_log_prob)`` per row and plots a scatter
    with a rolling-median overlay.
    """
    df = per_pdb_df.copy()
    df["perplexity"] = np.exp(-df["mean_log_prob"])
    df = df.sort_values("num_residues")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.scatter(
        df["num_residues"], df["perplexity"],
        s=14, alpha=0.5, color="steelblue",
        edgecolor="black", linewidth=0.2,
    )
    window = max(10, len(df) // 20)
    if len(df) >= window:
        rolling = df["perplexity"].rolling(window=window, center=True).median()
        ax.plot(df["num_residues"], rolling, color="tomato", linewidth=1.8,
                label=f"Rolling median (w={window})")
        ax.legend()

    ax.set_xlabel("Residues in structure")
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("Per-PDB perplexity vs structure size")
    _save(fig, path)
