"""Adapter tests: a real checkpoint load + one CPU inference on a fixture.

These are integration-flavoured (they download/load the canonical checkpoint),
so they're guarded behind an opt-in env var to keep the default ``make test``
suite fast and offline. Run with::

    UMA_RUN_SERVING_INTEGRATION=1 uv run pytest tests/test_serving/test_inference.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.serving.inference import InferenceEngine, InputTooLargeError
from src.serving.schemas import InferenceResult

FIXTURE_PDB = Path(__file__).parent.parent / "fixtures" / "1bc8.pdb"

_RUN_INTEGRATION = os.environ.get("UMA_RUN_SERVING_INTEGRATION") == "1"
integration = pytest.mark.skipif(
    not _RUN_INTEGRATION,
    reason="set UMA_RUN_SERVING_INTEGRATION=1 to run (loads the real checkpoint)",
)


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    # Generous cap so 1bc8 (~140 residues) is accepted in the test.
    return InferenceEngine(max_residues=10_000)


@integration
def test_run_inference_shapes_and_bounds(engine: InferenceEngine) -> None:
    pdb_text = FIXTURE_PDB.read_text(encoding="utf-8")
    result = engine.run(pdb_text, n_samples=2, temperature=0.1, seed=0)

    assert isinstance(result, InferenceResult)
    assert result.n_residues > 0
    assert len(result.sequences) == 2
    assert len(result.per_residue_confidence) == 2

    for seq, conf in zip(result.sequences, result.per_residue_confidence):
        assert len(seq) == result.n_residues
        assert len(conf) == result.n_residues
        assert all(0.0 <= c <= 1.0 for c in conf)

    assert 0.0 < result.mean_confidence <= 1.0
    assert result.inference_ms > 0.0


@integration
def test_score_native_sequence(engine: InferenceEngine) -> None:
    pdb_text = FIXTURE_PDB.read_text(encoding="utf-8")
    result = engine.score(pdb_text, mode="autoregressive", num_batches=2)

    assert result.n_residues > 0
    assert len(result.positions) == result.n_residues
    assert result.perplexity > 0.0
    assert 0.0 <= result.recovery <= 1.0
    assert len(result.sequence_scored) == result.n_residues
    for p in result.positions:
        assert p.top_aa  # model's preferred residue is populated
        assert 0.0 <= p.prob <= 1.0
        assert 0.0 <= p.top_prob <= 1.0
    assert result.inference_ms > 0.0


@integration
def test_residue_cap_rejects_oversized(engine: InferenceEngine) -> None:
    pdb_text = FIXTURE_PDB.read_text(encoding="utf-8")
    original = engine.max_residues
    engine.max_residues = 10  # 1bc8 has ~140 residues → over the cap
    try:
        with pytest.raises(InputTooLargeError):
            engine.run(pdb_text, n_samples=1)
    finally:
        engine.max_residues = original
