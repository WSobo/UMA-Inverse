"""Figure 4: training curves for the v2 3-stage curriculum.

Six panels (3 stages × 2 metrics: val_acc, val_loss). Read from the
PyTorch-Lightning CSVLogger metrics CSVs.

Sources:
    Stage 1: logs/csv/pairmixerinv-v2-stage1-nodes64/version_3/metrics.csv
    Stage 2: logs/csv/pairmixerinv-v2-stage2-nodes128-ddp4/version_1/metrics.csv
    Stage 3: logs/csv/pairmixerinv-v2-stage3-nodes384-ddp8/version_0/metrics.csv

(version numbers reflect re-runs after the EarlyStopping fix during training.)
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

STAGES = (
    ("stage1-nodes64",          "version_3", "Stage 1 — single A5500 (max nodes 64)",        15),
    ("stage2-nodes128-ddp4",    "version_1", "Stage 2 — 4× A5500 DDP (max nodes 128)",       25),
    ("stage3-nodes384-ddp8",    "version_0", "Stage 3 — 8× A5500 DDP (max nodes 384)",       30),
)


def _load_val_curve(metrics_csv: Path) -> tuple[list[int], list[float], list[float]]:
    """Return (epochs, val_acc, val_loss) — only rows where val/acc is populated."""
    if not metrics_csv.exists():
        return [], [], []
    epochs: list[int] = []
    val_acc: list[float] = []
    val_loss: list[float] = []
    with metrics_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            v_acc = row.get("val/acc", "")
            v_loss = row.get("val/loss", "")
            if not v_acc:
                continue
            try:
                ep = int(row.get("epoch", -1))
                acc = float(v_acc)
                loss = float(v_loss) if v_loss else float("nan")
            except ValueError:
                continue
            epochs.append(ep)
            val_acc.append(acc)
            val_loss.append(loss)
    return epochs, val_acc, val_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "figures",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=False)

    for col, (stage_name, version, title, max_epochs) in enumerate(STAGES):
        metrics_csv = (args.logs_dir / f"pairmixerinv-v2-{stage_name}" / version / "metrics.csv")
        epochs, val_acc, val_loss = _load_val_curve(metrics_csv)
        ax_acc = axes[0, col]
        ax_loss = axes[1, col]

        if epochs:
            ax_acc.plot(epochs, val_acc, marker="o", color="#2C5F8E", linewidth=1.5,
                        markersize=4, markerfacecolor="white", markeredgewidth=1.2)
            best_acc_idx = int(np.argmax(val_acc))
            ax_acc.plot(epochs[best_acc_idx], val_acc[best_acc_idx],
                        marker="*", color="#C13C3C", markersize=12, zorder=10)
            ax_acc.text(epochs[best_acc_idx], val_acc[best_acc_idx] + 0.01,
                        f"best {val_acc[best_acc_idx]:.3f}", color="#C13C3C",
                        ha="center", fontsize=8)

            ax_loss.plot(epochs, val_loss, marker="o", color="#2C5F8E", linewidth=1.5,
                         markersize=4, markerfacecolor="white", markeredgewidth=1.2)
            best_loss_idx = int(np.argmin(val_loss))
            ax_loss.plot(epochs[best_loss_idx], val_loss[best_loss_idx],
                         marker="*", color="#C13C3C", markersize=12, zorder=10)
            ax_loss.text(epochs[best_loss_idx], val_loss[best_loss_idx] - 0.05,
                         f"best {val_loss[best_loss_idx]:.3f}", color="#C13C3C",
                         ha="center", fontsize=8)

        ax_acc.set_title(title, fontsize=10)
        ax_acc.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax_acc.set_axisbelow(True)
        ax_acc.set_xlim(-0.5, max_epochs + 0.5)
        if col == 0:
            ax_acc.set_ylabel("val accuracy", fontsize=10)

        ax_loss.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax_loss.set_axisbelow(True)
        ax_loss.set_xlim(-0.5, max_epochs + 0.5)
        ax_loss.set_xlabel("epoch", fontsize=10)
        if col == 0:
            ax_loss.set_ylabel("val loss", fontsize=10)

    fig.suptitle(
        "v2 3-stage curriculum: val accuracy and val loss per epoch (red star = best per stage)",
        fontsize=11, y=1.0,
    )
    plt.tight_layout()

    pdf = args.out_dir / "fig4_training.pdf"
    png = args.out_dir / "fig4_training.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
