"""Pydantic V2 request/response contracts for the serving layer.

These models *are* the agent-readable API contract — FastAPI turns them into
the OpenAPI schema at ``/docs``. Field bounds double as the input-validation
and graceful-degradation guarantees the service advertises.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# ── Serving limits (single source of truth) ───────────────────────────────────
# A PDB this large is almost certainly a mistake or an abuse attempt; reject it
# at the field level before any parsing/inference work happens. 2 MB comfortably
# fits the bundled fixtures (1bc8 ≈ 141 KB, 4gyt ≈ 519 KB).
MAX_PDB_CHARS = 2_000_000
# Hard cap on samples per request — each sample is a full autoregressive decode
# (one decoder pass per residue), so this directly bounds CPU work.
MAX_N_SAMPLES = 8


class DesignRequest(BaseModel):
    """Input to ``POST /design``."""

    pdb: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PDB_CHARS,
        description="Full PDB file contents as text. Ligand atoms are read "
        "from the structure's HETATM records.",
    )
    ligand: str | None = Field(
        None,
        description="Reserved for forward-compatibility. The model conditions "
        "on ligand atoms parsed directly from the PDB HETATM records; this "
        "field does not inject a separate ligand.",
    )
    temperature: float = Field(
        0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 = argmax (most likely residue).",
    )
    n_samples: int = Field(
        1,
        ge=1,
        le=MAX_N_SAMPLES,
        description="Number of independent sequences to design for the structure.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"pdb": "HEADER ...\nATOM ...\n", "temperature": 0.1, "n_samples": 1}
            ]
        }
    }


class InferenceResult(BaseModel):
    """Output of the inference adapter and ``POST /design`` body.

    ``per_residue_confidence`` is one list per sample (softmax max-probability
    at each designable position). ``mean_confidence`` is the model's own
    LigandMPNN-style ``overall_confidence`` averaged across samples — surfaced,
    not recalibrated.
    """

    sequences: list[str] = Field(..., description="Designed amino-acid sequences (one per sample).")
    per_residue_confidence: list[list[float]] = Field(
        ...,
        description="Per-sample, per-residue softmax max-probability in [0, 1].",
    )
    mean_confidence: float = Field(..., description="Mean confidence across samples in (0, 1].")
    n_residues: int = Field(..., description="Number of residues in the parsed structure.")
    inference_ms: float = Field(..., description="Wall-clock decode time in milliseconds.")


class DesignResponse(InferenceResult):
    """``POST /design`` response: the inference result plus a request id."""

    request_id: str = Field(..., description="Unique id for this request (also in the X-Request-ID header).")


class HealthResponse(BaseModel):
    """Output of ``GET /health`` — fast liveness probe, no inference."""

    status: str = Field(..., description="'ok' once the model is loaded, else 'starting'.")
    model_loaded: bool
    uptime_s: float


class ErrorResponse(BaseModel):
    """Structured error body returned for 4xx/5xx (never a raw stack trace)."""

    error: str = Field(..., description="Machine-readable error category.")
    detail: str = Field(..., description="Human-readable explanation.")
    request_id: str | None = None
