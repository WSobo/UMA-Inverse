"""Benchmark sweeps: ligand-context ablation and temperature/diversity curve.

These read-only analyses take the model + validation split and produce
tables that answer the two questions most likely to land in a paper's
Methods section:

1. *Does ligand conditioning actually matter?* (Ablation — run the same
   eval with and without ligand features.)
2. *What's the diversity / quality trade-off from sampling?*
   (Temperature sweep — sample multiple sequences at increasing T,
   measure recovery + Hamming diversity.)

Both sweeps are slower than the single-pass teacher-forced evaluation in
:mod:`src.benchmarks.evaluation`, so they accept the same ``n_pdbs``
cap. Results are returned as lists of dataclasses; the aggregator in
:mod:`src.benchmarks.report` turns them into CSVs and figures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.benchmarks.evaluation import evaluate_validation_set
from src.benchmarks.metrics import hamming_diversity, recovery_rate
from src.data.ligandmpnn_bridge import load_json_ids, resolve_pdb_path
from src.inference.constraints import DesignConstraints
from src.inference.decoding import autoregressive_design
from src.inference.session import InferenceSession

logger = logging.getLogger(__name__)


# ─── Ligand-context ablation ──────────────────────────────────────────────────


@dataclass
class LigandAblationRow:
    """One row in ``ablation_ligand.csv`` — per-PDB ligand-aware vs masked."""

    pdb_id: str
    num_residues: int
    recovery_with_ligand: float
    recovery_masked: float
    delta_recovery: float  # with − masked
    mean_log_prob_with_ligand: float
    mean_log_prob_masked: float
    delta_log_prob: float


def run_ligand_ablation(
    session: InferenceSession,
    val_json: Path | str,
    pdb_dir: Path | str,
    *,
    n_pdbs: int | None = None,
    max_total_nodes: int | None = None,
    seed: int = 0,
    progress_callback=None,
) -> list[LigandAblationRow]:
    """Evaluate the val set twice (with / without ligand) and pair the results.

    Runs each configuration through :func:`evaluate_validation_set`
    separately, then joins on ``pdb_id`` so a single PDB that failed on
    one pass is dropped from both. Returns one row per PDB that survived
    both passes.
    """
    logger.info("ligand ablation: pass 1/2 (ligand enabled)")
    with_ligand = evaluate_validation_set(
        session, val_json, pdb_dir,
        n_pdbs=n_pdbs, mask_ligand=False,
        max_total_nodes=max_total_nodes, seed=seed,
        progress_callback=progress_callback,
    )
    logger.info("ligand ablation: pass 2/2 (ligand masked)")
    masked = evaluate_validation_set(
        session, val_json, pdb_dir,
        n_pdbs=n_pdbs, mask_ligand=True,
        max_total_nodes=max_total_nodes, seed=seed,
        progress_callback=progress_callback,
    )

    masked_by_id = {m.pdb_id: m for m in masked}
    rows: list[LigandAblationRow] = []
    for entry in with_ligand:
        mm = masked_by_id.get(entry.pdb_id)
        if mm is None:
            continue
        rows.append(
            LigandAblationRow(
                pdb_id=entry.pdb_id,
                num_residues=entry.num_residues,
                recovery_with_ligand=entry.recovery,
                recovery_masked=mm.recovery,
                delta_recovery=entry.recovery - mm.recovery,
                mean_log_prob_with_ligand=entry.mean_log_prob,
                mean_log_prob_masked=mm.mean_log_prob,
                delta_log_prob=entry.mean_log_prob - mm.mean_log_prob,
            )
        )
    logger.info("ligand ablation: paired %d PDBs", len(rows))
    return rows


# ─── Temperature / diversity sweep ────────────────────────────────────────────


@dataclass
class TemperatureRow:
    """One row in ``temperature_sweep.csv`` — stats for one T across all PDBs."""

    temperature: float
    top_p: float | None
    num_pdbs: int
    num_samples_per_pdb: int
    mean_recovery: float
    std_recovery: float
    mean_hamming_diversity: float   # between samples of the same PDB
    mean_overall_confidence: float
    mean_log_prob: float            # Per-position mean log-prob of sampled tokens


def run_temperature_sweep(
    session: InferenceSession,
    val_json: Path | str,
    pdb_dir: Path | str,
    *,
    temperatures: list[float],
    num_samples_per_pdb: int = 3,
    top_p: float | None = None,
    n_pdbs: int | None = None,
    max_total_nodes: int | None = None,
    seed: int = 0,
    progress_callback=None,
) -> list[TemperatureRow]:
    """Sample ``num_samples_per_pdb`` sequences at each temperature.

    Recovery is computed against the native sequence; diversity is the
    mean pairwise Hamming distance between samples of the same PDB.

    Sampling is autoregressive — this is the slowest benchmark, roughly
    ``num_samples_per_pdb × len(temperatures) × n_pdbs`` forward passes.
    """
    ids = load_json_ids(str(val_json))
    if n_pdbs is not None and n_pdbs < len(ids):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(ids), size=n_pdbs, replace=False)
        ids = [ids[int(i)] for i in sorted(chosen)]

    paths: list[Path] = []
    for pdb_id in ids:
        p = resolve_pdb_path(str(pdb_dir), pdb_id)
        if p is not None:
            paths.append(Path(p))

    rows: list[TemperatureRow] = []
    unconstrained = DesignConstraints.from_cli()

    for t_idx, temp in enumerate(temperatures):
        recoveries: list[float] = []
        diversities: list[float] = []
        confidences: list[float] = []
        log_probs: list[float] = []

        for p_idx, path in enumerate(paths):
            if progress_callback is not None:
                progress_callback(
                    t_idx * len(paths) + p_idx,
                    len(temperatures) * len(paths),
                    f"T={temp}:{path.stem}",
                )

            try:
                ctx = session.load_structure(path, max_total_nodes=max_total_nodes)
            except Exception as exc:
                logger.warning("skipping %s at T=%s: %s", path, temp, exc)
                continue
            resolved = unconstrained.resolve(ctx)
            samples = autoregressive_design(
                session=session,
                ctx=ctx,
                constraints=resolved,
                num_samples=num_samples_per_pdb,
                batch_size=num_samples_per_pdb,
                temperature=float(temp),
                top_p=top_p,
                seed=int(seed + p_idx),
            )

            native = ctx.native_sequence.cpu()
            for s in samples:
                recoveries.append(recovery_rate(s.token_ids, native))
                log_probs.append(
                    float(s.log_probs[resolved.designable_mask.cpu()].mean().item())
                )
                confidences.append(s.overall_confidence(resolved.designable_mask.cpu()))

            if len(samples) > 1:
                diversities.append(
                    hamming_diversity([s.token_ids for s in samples])
                )

        if not recoveries:
            logger.warning("no samples for T=%s", temp)
            continue

        rows.append(
            TemperatureRow(
                temperature=float(temp),
                top_p=top_p,
                num_pdbs=len(paths),
                num_samples_per_pdb=num_samples_per_pdb,
                mean_recovery=float(np.mean(recoveries)),
                std_recovery=float(np.std(recoveries)),
                mean_hamming_diversity=float(np.mean(diversities)) if diversities else 0.0,
                mean_overall_confidence=float(np.mean(confidences)),
                mean_log_prob=float(np.mean(log_probs)),
            )
        )
    return rows


# ─── Shared utility: timing reporter ──────────────────────────────────────────


def format_timing(elapsed_seconds: float) -> str:
    """Humanise a seconds count for log output (``"3.2s"`` / ``"1m 45s"``)."""
    if elapsed_seconds < 60:
        return f"{elapsed_seconds:.1f}s"
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    return f"{minutes}m {seconds}s"
