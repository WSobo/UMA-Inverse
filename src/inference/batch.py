"""Batch-mode driver: design sequences for many PDBs in one invocation.

The batch file format is a single JSON object mapping each PDB path to an
optional per-PDB constraint override block::

    {
        "inputs/1BC8.pdb": {},
        "inputs/4GYT.pdb": {
            "fix": "A12 A13",
            "bias": "W:3.0"
        }
    }

Override keys mirror :meth:`DesignConstraints.from_cli` — ``fix``,
``redesign``, ``design_chains``, ``parse_chains``, ``bias``, ``bias_file``,
``omit``, ``omit_file``, ``tie``, ``tie_weights``, ``mask_ligand``,
``include_zero_occupancy``, ``ligand_cutoff``, ``max_total_nodes``. Any
key that's also set at the CLI level applies as a global default; per-PDB
entries take precedence.

A ``.done.txt`` file is written alongside the run directory, one
completed PDB path per line. ``--resume`` skips PDBs already listed there.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PathLike = Path

_OVERRIDE_KEYS = frozenset(
    {
        "fix",
        "redesign",
        "design_chains",
        "parse_chains",
        "bias",
        "bias_file",
        "omit",
        "omit_file",
        "tie",
        "tie_weights",
        "mask_ligand",
        "include_zero_occupancy",
        "ligand_cutoff",
        "max_total_nodes",
    }
)


# ─── Batch spec loader ────────────────────────────────────────────────────────


@dataclass
class BatchEntry:
    """One PDB's worth of per-entry settings."""

    pdb_path: Path
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def pdb_id(self) -> str:
        return self.pdb_path.stem


def load_batch_spec(path: PathLike) -> list[BatchEntry]:
    """Parse the per-PDB JSON into :class:`BatchEntry` records.

    Raises :class:`ValueError` with a pointed message when the format is
    wrong. PDB paths are resolved relative to the spec file's directory
    when not absolute — mirroring LigandMPNN's pdb_path_multi convention.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"batch spec not found: {path}")

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level JSON object")

    entries: list[BatchEntry] = []
    base_dir = path.parent

    for key, value in raw.items():
        pdb_path = Path(key)
        if not pdb_path.is_absolute():
            pdb_path = (base_dir / pdb_path).resolve()

        overrides: dict[str, Any] = {}
        if value not in (None, "", {}):
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}: entry for {key!r} must be a mapping or empty; got {type(value).__name__}"
                )
            unknown = set(value) - _OVERRIDE_KEYS
            if unknown:
                raise ValueError(
                    f"{path}: unknown override keys for {key!r}: {sorted(unknown)} "
                    f"(allowed: {sorted(_OVERRIDE_KEYS)})"
                )
            overrides = dict(value)

        entries.append(BatchEntry(pdb_path=pdb_path, overrides=overrides))

    return entries


# ─── Resume support ───────────────────────────────────────────────────────────


def load_done_set(path: PathLike) -> set[str]:
    """Read the ``.done.txt`` file if present, returning the set of completed PDB paths."""
    path = Path(path)
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_done(path: PathLike, pdb_path: str) -> None:
    """Append one completed PDB path to the resume log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(pdb_path + "\n")


def filter_pending(
    entries: Iterable[BatchEntry],
    *,
    done_path: PathLike,
    resume: bool,
) -> list[BatchEntry]:
    """Return entries whose PDB path has not yet been recorded in ``done_path``.

    When ``resume`` is ``False`` the set is returned unchanged.
    """
    if not resume:
        return list(entries)
    done = load_done_set(done_path)
    if not done:
        return list(entries)
    remaining: list[BatchEntry] = []
    skipped = 0
    for entry in entries:
        if str(entry.pdb_path) in done:
            skipped += 1
            continue
        remaining.append(entry)
    if skipped:
        logger.info("resume: skipping %d PDB(s) recorded in %s", skipped, done_path)
    return remaining


# ─── Constraint merging ───────────────────────────────────────────────────────


def merge_constraint_kwargs(
    cli_defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Layer per-PDB overrides on top of CLI defaults.

    Overrides replace (not merge) — if an entry sets ``fix: "A1"``, any
    CLI-level ``--fix`` is ignored for that PDB. This mirrors LigandMPNN's
    behaviour where ``fix_residues_multi`` supersedes ``fixed_residues``.
    """
    merged = dict(cli_defaults)
    for key, value in overrides.items():
        merged[key] = value
    return merged
