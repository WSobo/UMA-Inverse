"""Production inference package for UMA-Inverse.

This package is the supported entry point for running sequence design and
scoring against a trained checkpoint. The layout separates concerns for
testability:

* :mod:`src.inference.session` — owns checkpoint + structure encoding.
* :mod:`src.inference.constraints` — residue-id parsing and design constraints.
* :mod:`src.inference.decoding` — sampling, autoregressive generation, scoring.
* :mod:`src.inference.output` — FASTA, JSON, and npz serialisation.
* :mod:`src.inference.cli` — Typer app exposing ``uma-inverse design / score``.

Programmatic use::

    from src.inference import InferenceSession, DesignConstraints, run_design

Preferred command-line use::

    uv run uma-inverse design --pdb structure.pdb --ckpt checkpoints/last.ckpt
"""
from src.inference.constraints import (
    ConstraintError,
    DesignConstraints,
    ResolvedConstraints,
    parse_residue_selection,
)
from src.inference.session import InferenceSession, StructureContext

__all__ = [
    "ConstraintError",
    "DesignConstraints",
    "InferenceSession",
    "ResolvedConstraints",
    "StructureContext",
    "parse_residue_selection",
]
