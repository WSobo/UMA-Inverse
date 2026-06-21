"""API contract tests for the FastAPI serving layer.

These run offline: the inference call is monkeypatched so no checkpoint is
loaded. They assert the request/response contract, error codes, headers, and
that metrics increment on real requests.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.serving import app as app_module  # noqa: E402
from src.serving.inference import InputTooLargeError  # noqa: E402
from src.serving.schemas import InferenceResult, ScorePosition, ScoreResult  # noqa: E402

_FAKE = InferenceResult(
    sequences=["ACDEFGHIKL"],
    per_residue_confidence=[[0.9] * 10],
    mean_confidence=0.9,
    n_residues=10,
    inference_ms=12.3,
)

_FAKE_SCORE = ScoreResult(
    positions=[
        ScorePosition(
            position=0, residue_id="A1", aa="K", log_prob=-2.3, prob=0.10,
            top_aa="R", top_prob=0.42,
        )
    ],
    mean_log_prob=-2.3,
    perplexity=9.97,
    recovery=0.0,
    n_residues=1,
    mode="autoregressive",
    use_sequence=True,
    num_batches=10,
    sequence_scored="K",
    inference_ms=20.0,
)

# A minimal well-formed body (pdb content is irrelevant — inference is mocked).
_BODY = {"pdb": "ATOM      1  N   MET A   1      0.0 0.0 0.0", "n_samples": 1}


@pytest.fixture
def client() -> TestClient:
    # No `with` block → the lifespan (real model load) does not run.
    return TestClient(app_module.app)


def test_design_valid_returns_200_and_schema(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_inference", lambda *a, **k: _FAKE)
    resp = client.post("/design", json=_BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["sequences"] == ["ACDEFGHIKL"]
    assert body["n_residues"] == 10
    assert body["mean_confidence"] == pytest.approx(0.9)
    assert "request_id" in body and body["request_id"]
    # Headers
    assert resp.headers["X-Request-ID"] == body["request_id"]
    assert float(resp.headers["X-Inference-MS"]) == pytest.approx(12.3, abs=0.01)


def test_design_oversized_returns_413(client, monkeypatch):
    def boom(*a, **k):
        raise InputTooLargeError(500, 120)

    monkeypatch.setattr(app_module, "run_inference", boom)
    resp = client.post("/design", json=_BODY)
    assert resp.status_code == 413
    assert resp.json()["error"] == "input_too_large"


def test_design_out_of_range_temperature_returns_422(client):
    resp = client.post("/design", json={"pdb": "ATOM ...", "temperature": 5.0})
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_design_missing_pdb_returns_422(client):
    resp = client.post("/design", json={"temperature": 0.1})
    assert resp.status_code == 422


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    # Lifespan didn't run, so no model is loaded in this test process.
    assert body["model_loaded"] is False
    assert body["status"] == "starting"
    assert body["uptime_s"] >= 0


def test_metrics_increment_after_request(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_inference", lambda *a, **k: _FAKE)
    client.post("/design", json=_BODY)
    text = client.get("/metrics").text
    assert "uma_requests_total" in text
    assert "uma_inference_latency_seconds" in text
    assert 'endpoint="/design"' in text


# ── /score ──────────────────────────────────────────────────────────────────


def test_score_valid_returns_200_and_schema(client, monkeypatch):
    monkeypatch.setattr(app_module, "score_inference", lambda *a, **k: _FAKE_SCORE)
    resp = client.post("/score", json={"pdb": "ATOM ...", "mode": "autoregressive"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["perplexity"] == pytest.approx(9.97)
    assert body["recovery"] == pytest.approx(0.0)
    assert body["positions"][0]["top_aa"] == "R"
    assert body["positions"][0]["aa"] == "K"
    assert "request_id" in body and body["request_id"]
    assert resp.headers["X-Request-ID"] == body["request_id"]
    assert float(resp.headers["X-Inference-MS"]) == pytest.approx(20.0, abs=0.01)


def test_score_bad_sequence_returns_400(client, monkeypatch):
    def boom(*a, **k):
        raise ValueError("sequence length 5 does not match the parsed residue count 46")

    monkeypatch.setattr(app_module, "score_inference", boom)
    resp = client.post("/score", json={"pdb": "ATOM ...", "sequence": "ACDEF"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad_request"


def test_score_oversized_returns_413(client, monkeypatch):
    def boom(*a, **k):
        raise InputTooLargeError(500, 256)

    monkeypatch.setattr(app_module, "score_inference", boom)
    resp = client.post("/score", json={"pdb": "ATOM ..."})
    assert resp.status_code == 413
    assert resp.json()["error"] == "input_too_large"


def test_score_num_batches_out_of_range_returns_422(client):
    resp = client.post("/score", json={"pdb": "ATOM ...", "num_batches": 999})
    assert resp.status_code == 422


def test_score_metrics_increment(client, monkeypatch):
    monkeypatch.setattr(app_module, "score_inference", lambda *a, **k: _FAKE_SCORE)
    client.post("/score", json={"pdb": "ATOM ..."})
    text = client.get("/metrics").text
    assert "uma_score_perplexity" in text
    assert 'endpoint="/score"' in text
