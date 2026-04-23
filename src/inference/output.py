"""Output serialisation: FASTA, manifests, per-residue confidence, and probabilities.

Every run writes:

* A FASTA per PDB with one record per sample (headers include
  ``overall_confidence``, ``ligand_confidence``, and the seed).
* A per-residue confidence JSON per PDB (entropy, top-3, margin).
* A single top-level ``run_manifest.json`` describing *how* the run was
  produced — git hash, checkpoint path, full config, seed, wall clock.

Optional outputs (toggled by caller):

* ``probs/<pdb>.npz`` — the full ``[num_samples, L, 21]`` probability
  tensor for downstream statistical analysis.
* ``ranked.csv`` — unique sequences across PDBs ranked by mean NLL.

Writing is kept separate from decoding so the library is usable in
contexts that don't want files on disk (e.g. notebook sessions calling
:func:`autoregressive_design` directly).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import socket
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.inference.decoding import DesignSample
from src.inference.session import StructureContext
from src.utils.io import ID_TO_AA

logger = logging.getLogger(__name__)
PathLike = os.PathLike

_MANIFEST_VERSION = "1"
_MAX_CONF_TOP_K = 3


# ─── Run manifest ─────────────────────────────────────────────────────────────


@dataclass
class RunManifest:
    """Lab-notebook-grade record of one inference run.

    Written once to ``<out_dir>/run_manifest.json`` at the start of a run
    (so crashes still leave a record) and updated with the stop timestamp
    on successful completion.
    """

    run_name: str
    command: str  # full CLI command, for quick grep-ability
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    config_path: str | None
    config_snapshot: dict[str, Any]
    git_hash: str
    hostname: str
    python_version: str
    torch_version: str
    model_revision: str  # git hash of the checkpoint-producing commit (if present)
    seed: int | None
    start_timestamp: str
    stop_timestamp: str | None = None
    num_pdbs: int = 0
    num_samples_per_pdb: int = 0
    temperature: float = 0.1
    top_p: float | None = None
    decoding_order: str = "random"
    extras: dict[str, Any] = field(default_factory=dict)
    manifest_version: str = _MANIFEST_VERSION

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True, default=str)

    def write(self, path: PathLike) -> None:
        Path(path).write_text(self.to_json())


def build_manifest(
    *,
    run_name: str,
    command: str,
    checkpoint_path: str | None,
    config_path: PathLike | None,
    config_snapshot: Mapping[str, Any],
    seed: int | None,
    temperature: float,
    top_p: float | None,
    decoding_order: str,
    num_pdbs: int,
    num_samples_per_pdb: int,
    extras: Mapping[str, Any] | None = None,
) -> RunManifest:
    """Capture every externally-visible input that shapes this run."""
    import sys

    ckpt_sha = _sha256_of(checkpoint_path) if checkpoint_path else None
    git_hash = _git_hash()
    model_revision = _embedded_git_hash(checkpoint_path) if checkpoint_path else "unknown"

    return RunManifest(
        run_name=run_name,
        command=command,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        checkpoint_sha256=ckpt_sha,
        config_path=str(config_path) if config_path else None,
        config_snapshot=dict(config_snapshot),
        git_hash=git_hash,
        hostname=socket.gethostname(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        model_revision=model_revision,
        seed=seed,
        start_timestamp=_now_iso(),
        num_pdbs=num_pdbs,
        num_samples_per_pdb=num_samples_per_pdb,
        temperature=temperature,
        top_p=top_p,
        decoding_order=decoding_order,
        extras=dict(extras or {}),
    )


# ─── FASTA writer ─────────────────────────────────────────────────────────────


def write_samples_fasta(
    path: PathLike,
    *,
    pdb_id: str,
    ctx: StructureContext,
    samples: Sequence[DesignSample],
    designable_mask: torch.Tensor,
    ligand_neighbour_mask: torch.Tensor | None = None,
    include_native: bool = True,
) -> None:
    """Write all samples for one PDB into a FASTA file.

    Headers follow LigandMPNN's schema so downstream tooling keeps working.
    ``overall_confidence`` / ``ligand_confidence`` match LigandMPNN's
    definitions exactly.

    Args:
        path: Destination FASTA path (parent directory must exist).
        pdb_id: Short identifier used as the primary header token.
        ctx: Structure context (for residue IDs and the native sequence).
        samples: One or more :class:`DesignSample` records to emit.
        designable_mask: Bool ``[L]`` — which positions count toward
            overall_confidence.
        ligand_neighbour_mask: Optional ``[L]`` — positions proximal to the
            ligand. When given, each header includes ``ligand_confidence``.
        include_native: When True, the first record is the parsed native
            sequence (tagged ``kind=native``).
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        if include_native:
            native_str = _ids_to_string(ctx.native_sequence.tolist())
            fh.write(f">{pdb_id} kind=native length={len(native_str)}\n{native_str}\n")

        for sample_idx, sample in enumerate(samples):
            seq_str = _ids_to_string(sample.token_ids.tolist())
            overall = sample.overall_confidence(designable_mask)
            header_parts = [
                pdb_id,
                f"sample={sample_idx}",
                f"seed={sample.seed}",
                f"T={sample.temperature}",
                f"overall_confidence={overall:.4f}",
            ]
            if sample.top_p is not None:
                header_parts.append(f"top_p={sample.top_p}")
            if ligand_neighbour_mask is not None:
                lig_conf = sample.ligand_confidence(designable_mask, ligand_neighbour_mask)
                header_parts.append(f"ligand_confidence={lig_conf:.4f}")
            header = " ".join(header_parts)
            fh.write(f">{header}\n{seq_str}\n")


# ─── Per-residue confidence JSON ──────────────────────────────────────────────


def write_per_residue_confidence(
    path: PathLike,
    *,
    pdb_id: str,
    ctx: StructureContext,
    samples: Sequence[DesignSample],
    designable_mask: torch.Tensor,
) -> None:
    """Emit per-residue confidence + top-K statistics.

    For every sample and every position we record:

    * ``top_k``: the top-3 AAs with probabilities.
    * ``entropy``: Shannon entropy of the full 21-token distribution.
    * ``margin``: probability gap between the best and second-best AA.
      Low margins flag positions where the model is torn between
      candidates — useful for prioritising experimental validation.
    * ``sampled``: the AA actually drawn and its probability.

    This is the piece LigandMPNN's ``overall_confidence`` aggregates; we
    surface the underlying distribution because it's cheap and widely
    useful downstream.
    """
    path = Path(path)
    payload: dict[str, Any] = {
        "pdb_id": pdb_id,
        "pdb_path": ctx.pdb_path,
        "num_residues": ctx.residue_count,
        "num_samples": len(samples),
        "residue_ids": ctx.residue_ids,
        "chain_ids": ctx.chain_ids,
        "designable": designable_mask.tolist(),
        "samples": [],
    }

    for sample in samples:
        probs = sample.probs_full  # [L, 21]
        tokens = sample.token_ids.tolist()
        entries: list[dict[str, Any]] = []
        for i in range(ctx.residue_count):
            row = probs[i]
            sampled_tok = int(tokens[i])
            sampled_prob = float(row[sampled_tok].item())
            top_k = torch.topk(row, k=min(_MAX_CONF_TOP_K, row.numel()))
            top_entries = [
                {"aa": ID_TO_AA.get(int(idx.item()), "X"), "prob": float(val.item())}
                for val, idx in zip(top_k.values, top_k.indices)
            ]
            if top_k.values.numel() > 1:
                margin = float((top_k.values[0] - top_k.values[1]).item())
            else:
                margin = 1.0
            entries.append(
                {
                    "position": i,
                    "residue_id": ctx.residue_ids[i],
                    "sampled": ID_TO_AA.get(sampled_tok, "X"),
                    "sampled_prob": sampled_prob,
                    "top_k": top_entries,
                    "entropy": _row_entropy(row),
                    "margin": margin,
                }
            )
        payload["samples"].append(
            {
                "seed": sample.seed,
                "temperature": sample.temperature,
                "top_p": sample.top_p,
                "overall_confidence": sample.overall_confidence(designable_mask),
                "positions": entries,
            }
        )

    path.write_text(json.dumps(payload, indent=2))


# ─── Probability tensor dump ──────────────────────────────────────────────────


def write_probs_npz(
    path: PathLike,
    *,
    pdb_id: str,
    ctx: StructureContext,
    samples: Sequence[DesignSample],
) -> None:
    """Persist the full ``[num_samples, L, 21]`` probability tensor as ``.npz``.

    Replaces LigandMPNN's pickle-based ``save_stats`` with a numpy-native
    format that doesn't require importing arbitrary Python classes.

    Keys written:

    * ``probs`` — ``[num_samples, L, 21]`` float32
    * ``token_ids`` — ``[num_samples, L]`` int64
    * ``log_probs`` — ``[num_samples, L]`` float32
    * ``decoding_order`` — ``[num_samples, L]`` int64
    * ``seeds`` — ``[num_samples]`` int64
    * ``native`` — ``[L]`` int64
    * ``residue_ids`` — ``[L]`` object (strings)
    * ``pdb_id`` — scalar str
    """
    path = Path(path)
    if not samples:
        raise ValueError("write_probs_npz called with empty sample list")

    probs = np.stack([s.probs_full.numpy() for s in samples]).astype(np.float32)
    token_ids = np.stack([s.token_ids.numpy() for s in samples]).astype(np.int64)
    log_probs = np.stack([s.log_probs.numpy() for s in samples]).astype(np.float32)
    decoding_order = np.stack([s.decoding_order.numpy() for s in samples]).astype(np.int64)
    seeds = np.array([s.seed for s in samples], dtype=np.int64)
    native = ctx.native_sequence.cpu().numpy().astype(np.int64)
    residue_ids = np.array(ctx.residue_ids, dtype=object)

    np.savez_compressed(
        path,
        probs=probs,
        token_ids=token_ids,
        log_probs=log_probs,
        decoding_order=decoding_order,
        seeds=seeds,
        native=native,
        residue_ids=residue_ids,
        pdb_id=str(pdb_id),
    )


# ─── Ranked CSV across runs ───────────────────────────────────────────────────


@dataclass
class RankedRow:
    """One row in ``ranked.csv`` — a unique designed sequence."""

    pdb_id: str
    sequence: str
    mean_nll: float
    overall_confidence: float
    ligand_confidence: float | None
    sample_seeds: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "sequence": self.sequence,
            "mean_nll": f"{self.mean_nll:.4f}",
            "overall_confidence": f"{self.overall_confidence:.4f}",
            "ligand_confidence": "" if self.ligand_confidence is None else f"{self.ligand_confidence:.4f}",
            "sample_seeds": ";".join(str(s) for s in self.sample_seeds),
        }


def build_ranked_rows(
    *,
    pdb_id: str,
    ctx: StructureContext,
    samples: Sequence[DesignSample],
    designable_mask: torch.Tensor,
    ligand_neighbour_mask: torch.Tensor | None = None,
) -> list[RankedRow]:
    """Dedupe samples by sequence and return rows sorted by mean NLL.

    Negative mean log-likelihood — lower is better. Samples with identical
    output sequences are merged; their seeds are concatenated so callers
    can reproduce any originating draw.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for sample in samples:
        seq = _ids_to_string(sample.token_ids.tolist())
        mean_lp = float(sample.log_probs[designable_mask.bool()].mean().item()) if designable_mask.any() else 0.0
        overall_conf = sample.overall_confidence(designable_mask)
        if ligand_neighbour_mask is not None:
            lig_conf = sample.ligand_confidence(designable_mask, ligand_neighbour_mask)
        else:
            lig_conf = None
        if seq in grouped:
            grouped[seq]["sample_seeds"].append(sample.seed)
        else:
            grouped[seq] = {
                "mean_nll": -mean_lp,
                "overall_confidence": overall_conf,
                "ligand_confidence": lig_conf,
                "sample_seeds": [sample.seed],
            }

    rows = [
        RankedRow(pdb_id=pdb_id, sequence=seq, **info)  # type: ignore[arg-type]
        for seq, info in grouped.items()
    ]
    rows.sort(key=lambda r: r.mean_nll)
    return rows


def write_ranked_csv(path: PathLike, rows: Iterable[RankedRow]) -> None:
    """Write the aggregated ranked rows to CSV (create or append)."""
    path = Path(path)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pdb_id",
                "sequence",
                "mean_nll",
                "overall_confidence",
                "ligand_confidence",
                "sample_seeds",
            ],
        )
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _ids_to_string(token_ids: Iterable[int]) -> str:
    return "".join(ID_TO_AA.get(int(idx), "X") for idx in token_ids)


def _row_entropy(row: torch.Tensor) -> float:
    """Shannon entropy (nats) of a single ``[21]`` probability row."""
    safe = row.clamp_min(1e-30)
    return float(-(safe * torch.log(safe)).sum().item())


def _sha256_of(path: PathLike | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_project_root(), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _embedded_git_hash(checkpoint_path: str) -> str:
    """Best-effort retrieval of the training-time git hash of a checkpoint.

    Training records the git hash under ``logs/runs/<run>/git_hash.txt``.
    When the checkpoint lives alongside such a file we surface it so the
    manifest can link the exact training commit to the produced weights.
    """
    try:
        from src.utils import io as _io_mod  # noqa: F401 (import check only)
    except ImportError:
        return "unknown"
    # The convention: checkpoints/<name>.ckpt → logs/runs/<latest-run>/git_hash.txt
    # We can't recover the run name from the ckpt path alone, so look for a
    # neighbouring metadata file first.
    parent = Path(checkpoint_path).parent
    for candidate in (parent / "git_hash.txt", parent / ".." / "git_hash.txt"):
        if candidate.exists():
            try:
                return candidate.read_text().strip()
            except OSError:
                continue
    return "unknown"


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ligand_neighbour_mask_from_ctx(ctx: StructureContext, cutoff: float) -> torch.Tensor:
    """Return a ``[L]`` bool mask of residues within ``cutoff`` Å of any ligand atom.

    When the structure has no ligand atoms this returns an all-False mask,
    which trips :meth:`DesignSample.ligand_confidence` to return 0.
    """
    L = ctx.residue_count
    N = ctx.z.shape[1]
    if N <= L:
        return torch.zeros(L, dtype=torch.bool, device=ctx.device)
    # Reconstruct residue-ligand distance mask from the encoded pair mask.
    # ctx.pair_mask[0, :L, L:] is True where residue i can attend to ligand j.
    # We don't have raw coords after encoding, so use the pair mask as a
    # proxy — any residue with an attending ligand atom is "neighbouring".
    # For an exact Å cutoff we'd need the raw coords; callers pass this at
    # load time instead when precision matters.
    return ctx.pair_mask[0, :L, L:].any(dim=-1)
