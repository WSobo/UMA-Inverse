"""Benchmarking and paper-grade evaluation for UMA-Inverse.

This package generates every metric and figure reported in the model
paper. Entry point is :func:`run_benchmark` (Python) or
``uma-inverse benchmark`` (CLI); it writes a reproducible directory with:

- ``summary.md`` / ``summary.json`` — headline numbers in paper-ready form.
- ``per_pdb.csv``, ``per_position.parquet`` — raw data for custom analyses.
- ``confusion_matrix.csv``, ``calibration.csv``, ``temperature_sweep.csv``,
  ``ablation_ligand.csv``, ``aa_composition.csv`` — individual metric tables.
- ``figures/*.png`` — Matplotlib renderings of the main results.
- ``run_manifest.json`` — git hash, ckpt sha256, config snapshot.

Modules:

* :mod:`src.benchmarks.metrics` — pure-function computations (recovery,
  perplexity, confusion, calibration, diversity). Independent of I/O.
* :mod:`src.benchmarks.evaluation` — iterates over a validation split,
  collects per-position records.
* :mod:`src.benchmarks.sweeps` — ligand-context ablation and
  temperature/diversity sweep.
* :mod:`src.benchmarks.plots` — Matplotlib figures for the paper.
* :mod:`src.benchmarks.report` — aggregates per-PDB records into the
  summary tables + prose.
* :mod:`src.benchmarks.cli` — Typer subcommand wiring.
"""
from src.benchmarks.evaluation import PerPositionRecord, evaluate_validation_set
from src.benchmarks.metrics import (
    aa_composition,
    calibration_bins,
    confusion_matrix,
    entropy_per_position,
    hamming_diversity,
    per_aa_recovery,
    perplexity,
    recovery_rate,
)

__all__ = [
    "PerPositionRecord",
    "aa_composition",
    "calibration_bins",
    "confusion_matrix",
    "entropy_per_position",
    "evaluate_validation_set",
    "hamming_diversity",
    "per_aa_recovery",
    "perplexity",
    "recovery_rate",
]
