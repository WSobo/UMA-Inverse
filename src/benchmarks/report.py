"""Aggregate raw benchmark records into tables, figures, and a summary.

Takes the outputs of :mod:`src.benchmarks.evaluation` and
:mod:`src.benchmarks.sweeps` and produces:

* ``per_pdb.csv``, ``per_position.parquet`` — the raw tables every later
  analysis can re-aggregate.
* ``confusion_matrix.csv``, ``calibration.csv``, ``aa_composition.csv``,
  ``ablation_ligand.csv``, ``temperature_sweep.csv`` — metric-by-metric
  tables used by the paper.
* ``figures/*.png`` — :mod:`src.benchmarks.plots` outputs.
* ``summary.md`` / ``summary.json`` — headline numbers in Markdown for
  direct paste into a report and JSON for downstream parsing.

Callers produce the raw data, then call :func:`write_report`. Nothing in
this module touches the model.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.benchmarks import plots
from src.benchmarks.evaluation import PdbEvaluation
from src.benchmarks.metrics import (
    aa_composition,
    calibration_bins,
    confusion_matrix,
    confusion_matrix_normalized,
    expected_calibration_error,
    per_aa_recovery,
    perplexity,
    recovery_rate,
)
from src.benchmarks.sweeps import LigandAblationRow, TemperatureRow
from src.utils.io import ID_TO_AA

logger = logging.getLogger(__name__)

_AA_LETTERS = [ID_TO_AA[i] for i in range(20)]


# ─── Assembled tables ─────────────────────────────────────────────────────────


def per_position_frame(evaluations: Sequence[PdbEvaluation]) -> pd.DataFrame:
    """Stack every :class:`PerPositionRecord` into a single long table."""
    rows = []
    for ev in evaluations:
        for rec in ev.per_position:
            rows.append(
                {
                    "pdb_id": rec.pdb_id,
                    "residue_id": rec.residue_id,
                    "chain_id": rec.chain_id,
                    "position": rec.position,
                    "native_token": rec.native_token,
                    "pred_token": rec.pred_token,
                    "native_aa": ID_TO_AA.get(rec.native_token, "X"),
                    "pred_aa": ID_TO_AA.get(rec.pred_token, "X"),
                    "native_log_prob": rec.native_log_prob,
                    "entropy": rec.entropy,
                    "distance_to_ligand": rec.distance_to_ligand,
                    "ligand_context_masked": rec.ligand_context_masked,
                }
            )
    return pd.DataFrame.from_records(rows)


def per_pdb_frame(evaluations: Sequence[PdbEvaluation]) -> pd.DataFrame:
    """One row per PDB — recovery, perplexity, size, wall clock."""
    rows = [
        {
            "pdb_id": ev.pdb_id,
            "num_residues": ev.num_residues,
            "recovery": ev.recovery,
            "mean_log_prob": ev.mean_log_prob,
            "perplexity": float(np.exp(-ev.mean_log_prob))
            if np.isfinite(ev.mean_log_prob) else float("nan"),
            "wall_seconds": ev.wall_seconds,
        }
        for ev in evaluations
    ]
    return pd.DataFrame.from_records(rows)


# ─── Headline numbers for summary ─────────────────────────────────────────────


def compute_headline_stats(evaluations: Sequence[PdbEvaluation]) -> dict:
    """Headline numbers ready for inclusion in summary.md."""
    if not evaluations:
        return {"num_pdbs": 0}

    all_native = torch.cat([torch.as_tensor(e.native_sequence) for e in evaluations])
    all_pred = torch.cat([torch.as_tensor(e.pred_sequence) for e in evaluations])
    all_log_probs = torch.cat(
        [
            torch.tensor([r.native_log_prob for r in e.per_position], dtype=torch.float32)
            for e in evaluations
        ]
    )
    all_entropies = torch.cat(
        [
            torch.tensor([r.entropy for r in e.per_position], dtype=torch.float32)
            for e in evaluations
        ]
    )

    # Pool probability and correctness vectors for calibration
    prob_rows = []
    correct_rows = []
    for ev in evaluations:
        probs_full = ev.probs_full  # [L, 21]
        for rec in ev.per_position:
            i = rec.position
            prob_rows.append(float(probs_full[i, rec.pred_token]))
            correct_rows.append(rec.pred_token == rec.native_token)
    predicted_probs = torch.tensor(prob_rows, dtype=torch.float32)
    correct = torch.tensor(correct_rows, dtype=torch.bool)

    bins = calibration_bins(predicted_probs, correct, num_bins=10)
    ece = expected_calibration_error(bins)

    per_aa = per_aa_recovery(all_pred, all_native)
    native_comp = aa_composition(all_native)
    pred_comp = aa_composition(all_pred)

    per_pdb_recovery = np.array([ev.recovery for ev in evaluations])

    return {
        "num_pdbs": len(evaluations),
        "total_residues": int((all_native != 20).sum().item()),
        "overall_recovery": recovery_rate(all_pred, all_native),
        "mean_pdb_recovery": float(per_pdb_recovery.mean()),
        "std_pdb_recovery": float(per_pdb_recovery.std()),
        "median_pdb_recovery": float(np.median(per_pdb_recovery)),
        "perplexity": perplexity(all_log_probs),
        "mean_entropy": float(all_entropies.mean().item()),
        "expected_calibration_error": ece,
        "per_aa_recovery": per_aa,
        "native_aa_composition": native_comp,
        "predicted_aa_composition": pred_comp,
    }


# ─── Writers ──────────────────────────────────────────────────────────────────


def write_report(
    out_dir: Path,
    *,
    evaluations: Sequence[PdbEvaluation],
    ablation_rows: Sequence[LigandAblationRow] | None = None,
    temperature_rows: Sequence[TemperatureRow] | None = None,
    run_metadata: dict | None = None,
) -> None:
    """Write every table and figure for one benchmark run.

    ``run_metadata`` is an optional free-form dict (git hash, ckpt path,
    arguments) spliced into the summary for provenance.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pos_df = per_position_frame(evaluations)
    pdb_df = per_pdb_frame(evaluations)

    pos_df.to_parquet(out_dir / "per_position.parquet", index=False)
    pdb_df.to_csv(out_dir / "per_pdb.csv", index=False)

    headline = compute_headline_stats(evaluations)

    # ── Confusion matrix
    all_native = torch.cat([torch.as_tensor(e.native_sequence) for e in evaluations])
    all_pred = torch.cat([torch.as_tensor(e.pred_sequence) for e in evaluations])
    cm_counts = confusion_matrix(all_pred, all_native, num_classes=20)
    cm_norm = confusion_matrix_normalized(cm_counts, axis="native")
    pd.DataFrame(cm_counts, index=_AA_LETTERS, columns=_AA_LETTERS).to_csv(
        out_dir / "confusion_matrix.csv", index_label="native\\predicted"
    )

    # ── Calibration
    prob_rows = []
    correct_rows = []
    for ev in evaluations:
        for rec in ev.per_position:
            prob_rows.append(float(ev.probs_full[rec.position, rec.pred_token]))
            correct_rows.append(rec.pred_token == rec.native_token)
    cal_bins = calibration_bins(
        torch.tensor(prob_rows, dtype=torch.float32),
        torch.tensor(correct_rows, dtype=torch.bool),
        num_bins=10,
    )
    pd.DataFrame([asdict(b) for b in cal_bins]).to_csv(
        out_dir / "calibration.csv", index=False
    )

    # ── AA composition
    comp_df = pd.DataFrame(
        {
            "amino_acid": _AA_LETTERS,
            "native_freq": [headline["native_aa_composition"].get(a, 0.0) for a in _AA_LETTERS],
            "predicted_freq": [headline["predicted_aa_composition"].get(a, 0.0) for a in _AA_LETTERS],
            "per_aa_recovery": [headline["per_aa_recovery"].get(a, float("nan")) for a in _AA_LETTERS],
        }
    )
    comp_df.to_csv(out_dir / "aa_composition.csv", index=False)

    # ── Ablation
    if ablation_rows:
        pd.DataFrame([asdict(r) for r in ablation_rows]).to_csv(
            out_dir / "ablation_ligand.csv", index=False
        )

    # ── Temperature sweep
    if temperature_rows:
        pd.DataFrame([asdict(r) for r in temperature_rows]).to_csv(
            out_dir / "temperature_sweep.csv", index=False
        )

    # ── Figures
    plots.plot_confusion_matrix(
        cm_normalised=cm_norm,
        aa_labels=_AA_LETTERS,
        path=figures_dir / "confusion.png",
    )
    plots.plot_calibration(
        bins=cal_bins,
        ece=headline["expected_calibration_error"],
        path=figures_dir / "calibration.png",
    )
    plots.plot_near_ligand_recovery(
        per_position_df=pos_df,
        path=figures_dir / "near_ligand.png",
    )
    plots.plot_aa_composition(
        native_freqs=headline["native_aa_composition"],
        predicted_freqs=headline["predicted_aa_composition"],
        path=figures_dir / "aa_composition.png",
    )
    plots.plot_perplexity_by_length(
        per_pdb_df=pdb_df,
        path=figures_dir / "perplexity_by_length.png",
    )
    if temperature_rows:
        plots.plot_temperature_diversity(
            temperature_df=pd.DataFrame([asdict(r) for r in temperature_rows]),
            path=figures_dir / "temperature_diversity.png",
        )

    # ── Summary files
    _write_summary_json(
        out_dir / "summary.json",
        headline=headline,
        ablation=ablation_rows,
        temperature=temperature_rows,
        run_metadata=run_metadata,
    )
    _write_summary_md(
        out_dir / "summary.md",
        headline=headline,
        ablation=ablation_rows,
        temperature=temperature_rows,
        run_metadata=run_metadata,
    )


def _write_summary_json(
    path: Path,
    *,
    headline: dict,
    ablation: Sequence[LigandAblationRow] | None,
    temperature: Sequence[TemperatureRow] | None,
    run_metadata: dict | None,
) -> None:
    payload: dict = {"headline": headline}
    if ablation:
        a_df = pd.DataFrame([asdict(r) for r in ablation])
        payload["ablation_ligand"] = {
            "num_pdbs": len(a_df),
            "mean_recovery_with_ligand": float(a_df["recovery_with_ligand"].mean()),
            "mean_recovery_masked": float(a_df["recovery_masked"].mean()),
            "mean_delta_recovery": float(a_df["delta_recovery"].mean()),
            "mean_delta_log_prob": float(a_df["delta_log_prob"].mean()),
        }
    if temperature:
        payload["temperature_sweep"] = [asdict(r) for r in temperature]
    if run_metadata:
        payload["run_metadata"] = run_metadata
    path.write_text(json.dumps(payload, indent=2, default=str))


def _write_summary_md(
    path: Path,
    *,
    headline: dict,
    ablation: Sequence[LigandAblationRow] | None,
    temperature: Sequence[TemperatureRow] | None,
    run_metadata: dict | None,
) -> None:
    lines = [
        "# UMA-Inverse benchmark summary",
        "",
    ]
    if run_metadata:
        lines += [
            f"- **checkpoint**: `{run_metadata.get('checkpoint_path', '(none)')}`",
            f"- **git hash**: `{run_metadata.get('git_hash', 'unknown')}`",
            f"- **run timestamp**: {run_metadata.get('start_timestamp', 'unknown')}",
            "",
        ]
    lines += [
        "## Headline",
        "",
        f"- **PDBs evaluated**: {headline['num_pdbs']}",
        f"- **Residues evaluated**: {headline['total_residues']}",
        f"- **Pooled sequence recovery**: {headline['overall_recovery']:.4f}",
        f"- **Per-PDB recovery**: {headline['mean_pdb_recovery']:.4f} ± {headline['std_pdb_recovery']:.4f} "
        f"(median {headline['median_pdb_recovery']:.4f})",
        f"- **Perplexity**: {headline['perplexity']:.4f}",
        f"- **Mean entropy (nats)**: {headline['mean_entropy']:.4f}",
        f"- **Expected calibration error (ECE)**: {headline['expected_calibration_error']:.4f}",
        "",
    ]

    lines += ["## Per-AA recovery", "", "| AA | Recovery | Native freq | Predicted freq |", "|---|---|---|---|"]
    for aa in _AA_LETTERS:
        lines.append(
            f"| {aa} | {headline['per_aa_recovery'].get(aa, float('nan')):.3f} | "
            f"{headline['native_aa_composition'].get(aa, 0.0):.3f} | "
            f"{headline['predicted_aa_composition'].get(aa, 0.0):.3f} |"
        )
    lines.append("")

    if ablation:
        a_df = pd.DataFrame([asdict(r) for r in ablation])
        lines += [
            "## Ligand-context ablation",
            "",
            f"- **Paired PDBs**: {len(a_df)}",
            f"- **Recovery with ligand**: {a_df['recovery_with_ligand'].mean():.4f}",
            f"- **Recovery with ligand masked**: {a_df['recovery_masked'].mean():.4f}",
            f"- **Δ recovery**: {a_df['delta_recovery'].mean():.4f}",
            f"- **Δ mean log-prob**: {a_df['delta_log_prob'].mean():.4f}",
            "",
            "Positive Δ indicates ligand context improves prediction.",
            "",
        ]

    if temperature:
        lines += [
            "## Temperature sweep",
            "",
            "| T | Recovery (mean ± std) | Mean Hamming diversity | Mean overall conf |",
            "|---|---|---|---|",
        ]
        for r in temperature:
            lines.append(
                f"| {r.temperature} | {r.mean_recovery:.4f} ± {r.std_recovery:.4f} "
                f"| {r.mean_hamming_diversity:.4f} | {r.mean_overall_confidence:.4f} |"
            )
        lines.append("")

    lines += [
        "## Figures",
        "",
        "- ![Confusion matrix](figures/confusion.png)",
        "- ![Calibration](figures/calibration.png)",
        "- ![Recovery vs distance to ligand](figures/near_ligand.png)",
        "- ![AA composition](figures/aa_composition.png)",
        "- ![Perplexity vs length](figures/perplexity_by_length.png)",
    ]
    if temperature:
        lines.append("- ![Temperature vs diversity](figures/temperature_diversity.png)")
    lines.append("")

    path.write_text("\n".join(lines))
