"""Validation-set evaluation: teacher-forced scoring with full per-position output.

Given a trained checkpoint, a split JSON (``LigandMPNN/training/valid.json``
or test equivalents) and a PDB directory, this module:

1. Iterates over every PDB in the split that can be resolved.
2. Runs a single teacher-forced forward pass per PDB (left-to-right
   decoding order, native sequence in the AR context) — the standard
   "sequence recovery" evaluation protocol used in every inverse-folding
   paper.
3. Collects a :class:`PerPositionRecord` per residue with the native
   AA, the argmax prediction, the native log-probability, full entropy,
   and distance-to-nearest-ligand-atom.

The per-position records flow into :mod:`src.benchmarks.report` for
aggregation and figure production. They are *not* run through sampling
— sampling is the separate job of :mod:`src.benchmarks.sweeps` (for the
temperature/diversity curve).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.benchmarks.metrics import residue_ligand_distances
from src.data.ligandmpnn_bridge import load_json_ids, resolve_pdb_path
from src.data.pdb_parser import parse_pdb
from src.inference.session import InferenceSession, StructureContext

logger = logging.getLogger(__name__)


# ─── Per-position record ──────────────────────────────────────────────────────


@dataclass
class PerPositionRecord:
    """One evaluation row: one PDB, one residue."""

    pdb_id: str
    residue_id: str
    chain_id: str
    position: int          # Position index within the loaded structure (0..L-1)
    native_token: int      # 0..19 AA token (X excluded)
    pred_token: int        # argmax prediction
    native_log_prob: float  # log p(native | structure, native context)
    entropy: float          # Shannon entropy of the position's distribution (nats)
    distance_to_ligand: float  # Å to nearest ligand atom (inf if none)
    ligand_context_masked: bool  # True if --mask-ligand was applied


# ─── Per-PDB result ───────────────────────────────────────────────────────────


@dataclass
class PdbEvaluation:
    """Aggregate record returned by :func:`evaluate_single_pdb`."""

    pdb_id: str
    num_residues: int
    recovery: float              # argmax vs native; X excluded
    mean_log_prob: float         # -perplexity_in_nats_per_residue
    per_position: list[PerPositionRecord]
    # Keep the full [L, 21] distribution for the ligand-ablation delta plot —
    # aggregated later rather than per row (kept in memory as numpy to avoid
    # torch overhead in the report aggregator).
    probs_full: np.ndarray       # shape [L, 21]
    native_sequence: np.ndarray  # [L]
    pred_sequence: np.ndarray    # [L]
    wall_seconds: float


# ─── Single-PDB evaluation ────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_single_pdb(
    session: InferenceSession,
    pdb_path: Path | str,
    *,
    mask_ligand: bool = False,
    max_total_nodes: int | None = None,
) -> PdbEvaluation | None:
    """Teacher-forced evaluation of one PDB under the loaded model.

    Returns ``None`` if the structure fails to parse (caught and logged
    rather than raising — benchmarking over thousands of PDBs needs to be
    resilient to individual bad files).
    """
    pdb_path = Path(pdb_path)
    pdb_id = pdb_path.stem
    start = time.perf_counter()

    try:
        ctx = session.load_structure(
            pdb_path=pdb_path,
            mask_ligand=mask_ligand,
            max_total_nodes=max_total_nodes,
        )
    except Exception as exc:
        logger.warning("skipping %s: %s", pdb_path, exc)
        return None

    logits = _teacher_forced_logits(session, ctx)  # [L, 21]
    probs = torch.softmax(logits, dim=-1)
    log_probs_all = torch.log_softmax(logits, dim=-1)

    native = ctx.native_sequence  # [L]
    pred = logits.argmax(dim=-1)  # [L]

    safe_probs = probs.clamp_min(1e-30)
    entropy = -(safe_probs * torch.log(safe_probs)).sum(dim=-1)  # [L]

    native_log_prob = log_probs_all.gather(1, native.unsqueeze(-1)).squeeze(-1)  # [L]

    # Compute residue→ligand distance for the near-ligand breakdown.
    # We need the raw coords — pull them from the parser output rather than
    # the context (which keeps only encoded tensors).
    residue_coords, ligand_coords = _raw_coords_for_ctx(ctx)
    distances = residue_ligand_distances(residue_coords, ligand_coords)

    records: list[PerPositionRecord] = []
    for i in range(ctx.residue_count):
        nat = int(native[i].item())
        if nat == 20:
            # X residues are parseable but not meaningful for recovery — skip
            continue
        records.append(
            PerPositionRecord(
                pdb_id=pdb_id,
                residue_id=ctx.residue_ids[i],
                chain_id=ctx.chain_ids[i],
                position=i,
                native_token=nat,
                pred_token=int(pred[i].item()),
                native_log_prob=float(native_log_prob[i].item()),
                entropy=float(entropy[i].item()),
                distance_to_ligand=float(distances[i].item()),
                ligand_context_masked=mask_ligand,
            )
        )

    elapsed = time.perf_counter() - start

    # Recovery over real (non-X) residues only
    valid = native != 20
    if valid.any():
        recovery = float(((pred == native) & valid).sum().item()) / float(valid.sum().item())
        mean_lp = float(native_log_prob[valid].mean().item())
    else:
        recovery = 0.0
        mean_lp = float("nan")

    return PdbEvaluation(
        pdb_id=pdb_id,
        num_residues=ctx.residue_count,
        recovery=recovery,
        mean_log_prob=mean_lp,
        per_position=records,
        probs_full=probs.cpu().numpy(),
        native_sequence=native.cpu().numpy(),
        pred_sequence=pred.cpu().numpy(),
        wall_seconds=elapsed,
    )


def _teacher_forced_logits(session: InferenceSession, ctx: StructureContext) -> torch.Tensor:
    """Single forward pass with native sequence as AR context.

    This is the standard "sequence recovery" benchmark protocol: every
    position is evaluated in a fixed left-to-right decoding order with the
    full native sequence supplied as context. It is deterministic given
    the loaded weights; no random decoding orders to average.

    Returns:
        ``[L, 21]`` logit tensor; token 20 (X) is masked to ``-inf`` so
        argmax selections never return it.
    """
    model = session.model
    L = ctx.residue_count

    # Left-to-right decoding order: ranks[position] = position index
    ranks = torch.arange(L, device=ctx.device).unsqueeze(0)

    ar_ctx = model._autoregressive_context(
        z=ctx.z,
        sequence=ctx.native_sequence.unsqueeze(0),
        residue_mask=ctx.residue_mask,
        decoding_order=ranks,
    )
    decoder_input = torch.cat([ctx.node_repr_res, ar_ctx, ctx.lig_ctx], dim=-1)
    logits = model.decoder(decoder_input)[0]  # [L, 21]

    # Token 20 (X) is not a valid output — clamp so it never wins argmax
    logits = logits.clone()
    logits[:, 20] = float("-inf")
    return logits


def _raw_coords_for_ctx(ctx: StructureContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-parse the PDB to recover raw Cα / ligand coordinates.

    The :class:`StructureContext` holds only encoded features for space —
    distance-to-ligand needs the raw xyz values. Re-parsing is cheap
    (milliseconds per PDB) and avoids bloating the context dataclass.
    """
    parsed = parse_pdb(ctx.pdb_path)
    x = parsed["X"]
    mask = parsed["mask"].bool()
    residue_coords_all = x[:, 1, :][mask]  # Cα only, valid residues

    y = parsed["Y"]
    y_m = parsed["Y_m"].bool()
    ligand_coords_all = y[y_m]

    # Context may have cropped residues; align by taking the first
    # `ctx.residue_count` valid ones (matches the bridge's crop ordering).
    # For inference we typically pass max_total_nodes=very_high so the crop
    # is a no-op; this slice is defensive.
    if residue_coords_all.shape[0] >= ctx.residue_count:
        residue_coords = residue_coords_all[: ctx.residue_count]
    else:
        residue_coords = residue_coords_all

    return residue_coords, ligand_coords_all


# ─── Validation-set iterator ──────────────────────────────────────────────────


def evaluate_validation_set(
    session: InferenceSession,
    val_json: Path | str,
    pdb_dir: Path | str,
    *,
    n_pdbs: int | None = None,
    mask_ligand: bool = False,
    max_total_nodes: int | None = None,
    seed: int = 0,
    progress_callback=None,
) -> list[PdbEvaluation]:
    """Evaluate every resolvable PDB in a split.

    Args:
        session: Loaded inference session.
        val_json: Path to a LigandMPNN-style list of PDB IDs.
        pdb_dir: Directory tree containing the parsed PDB files.
        n_pdbs: Cap on number of PDBs to evaluate. ``None`` = all.
            A random subset (via ``seed``) is taken when capped so the
            sample is IID rather than alphabetical.
        mask_ligand: Zero ligand features before encoding — used by the
            ligand-context ablation.
        max_total_nodes: Override the config's crop budget. Benchmarking
            should use a large value (or keep config default) so
            structures aren't silently cropped.
        seed: RNG seed for subsampling reproducibility.
        progress_callback: Optional ``callable(idx, total, pdb_id)`` hook
            for CLI progress bars.

    Returns:
        List of :class:`PdbEvaluation` records, one per successfully
        evaluated PDB. PDBs that fail to parse or resolve are logged and
        skipped — the returned list may be shorter than ``n_pdbs``.
    """
    pdb_dir = str(pdb_dir)
    ids = load_json_ids(str(val_json))

    if n_pdbs is not None and n_pdbs < len(ids):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(ids), size=n_pdbs, replace=False)
        ids = [ids[int(i)] for i in sorted(chosen)]

    results: list[PdbEvaluation] = []
    skipped_missing = 0
    for idx, pdb_id in enumerate(ids):
        path = resolve_pdb_path(pdb_dir, pdb_id)
        if path is None:
            skipped_missing += 1
            continue
        if progress_callback is not None:
            progress_callback(idx, len(ids), pdb_id)

        result = evaluate_single_pdb(
            session=session,
            pdb_path=path,
            mask_ligand=mask_ligand,
            max_total_nodes=max_total_nodes,
        )
        if result is not None:
            results.append(result)

    if skipped_missing:
        logger.info("skipped %d PDB(s) with no file in %s", skipped_missing, pdb_dir)
    logger.info("evaluated %d/%d PDBs", len(results), len(ids))
    return results
