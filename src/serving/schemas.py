"""Pydantic V2 request/response contracts for the serving layer.

These models *are* the agent-readable API contract — FastAPI turns them into
the OpenAPI schema at ``/docs``. Field bounds double as the input-validation
and graceful-degradation guarantees the service advertises.
"""
from __future__ import annotations

from typing import Literal

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
        description="Full structure file contents as text — legacy PDB or mmCIF "
        "(.cif); the format is auto-detected. Ligand atoms are read from the "
        "structure's HETATM / _atom_site records.",
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

    # ── Advanced decoding controls (optional; defaults reproduce prior behaviour) ──
    top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus (top-p) sampling threshold. None disables it (sample "
        "from the full temperature-scaled distribution).",
    )
    seed: int | None = Field(
        None,
        ge=0,
        description="Base RNG seed for reproducibility. Sample i uses seed + i. "
        "None draws a fresh random seed each request.",
    )
    decoding_order: Literal["random", "left-to-right"] = Field(
        "random",
        description="Autoregressive decoding order. 'random' matches LigandMPNN; "
        "'left-to-right' is deterministic given a seed.",
    )

    # ── Advanced design constraints (optional; LigandMPNN-style selectors) ─────────
    fix: str | None = Field(
        None,
        description="Residue IDs to lock to their native identity, e.g. 'A23 A24 B10' "
        "or 'A23,A24,B10'. These positions are excluded from redesign.",
    )
    redesign: str | None = Field(
        None,
        description="If set, ONLY these residue IDs are designable (all others held "
        "native). Same selector syntax as 'fix'. Mutually exclusive per-residue with fix.",
    )
    design_chains: str | None = Field(
        None,
        description="Comma/space-separated chain letters to redesign, e.g. 'A,B'. "
        "Narrows the designable set to these chains.",
    )
    bias: str | None = Field(
        None,
        description="Global amino-acid logit bias, e.g. 'W:3.0,A:-1.0' (added to logits "
        "before softmax; positive favours, negative disfavours).",
    )
    omit: str | None = Field(
        None,
        description="Amino acids to forbid everywhere, as a letter string 'CDFG' or "
        "'C,D,F,G'.",
    )
    tie: str | None = Field(
        None,
        description="Symmetry tie groups — residues forced to the same identity. "
        "Groups separated by '|', members by ',', e.g. 'A1,B1|A5,B5'.",
    )
    tie_weights: str | None = Field(
        None,
        description="Optional per-member weights aligned to 'tie' (same '|'/',' layout). "
        "Defaults to equal weights within each group.",
    )
    mask_ligand: bool = Field(
        False,
        description="Ablation toggle: if true, hide ligand atoms from the model "
        "(design as if apo). Useful for A/B comparisons.",
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


class ScoreRequest(BaseModel):
    """Input to ``POST /score``."""

    pdb: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PDB_CHARS,
        description="Full structure file contents as text — legacy PDB or mmCIF "
        "(.cif), format auto-detected. Ligand atoms are read from HETATM / _atom_site records.",
    )
    sequence: str | None = Field(
        None,
        description="Optional amino-acid sequence (one-letter codes) to score against "
        "the structure. Must match the parsed residue count. Defaults to the PDB's "
        "native sequence.",
    )
    mode: Literal["autoregressive", "single-aa"] = Field(
        "autoregressive",
        description="'autoregressive' averages log-probs over random decoding orders "
        "(num_batches forward passes — fast). 'single-aa' scores each residue with all "
        "others visible (one pass per residue — slower).",
    )
    use_sequence: bool = Field(
        True,
        description="If False, the decoder sees only structural context (sequence masked) "
        "— measures how much signal the sequence itself adds.",
    )
    num_batches: int = Field(
        10, ge=1, le=20, description="Random decoding orders to average (autoregressive mode)."
    )


class ScorePosition(BaseModel):
    """Per-residue scoring result, including the model's preferred residue."""

    position: int
    residue_id: str = Field(..., description="Chain+number identifier, e.g. 'A23'.")
    aa: str = Field(..., description="The scored residue (native or provided).")
    log_prob: float = Field(..., description="Log-probability of the scored residue.")
    prob: float = Field(..., description="exp(log_prob) — probability of the scored residue.")
    top_aa: str = Field(..., description="The model's most-likely residue at this position.")
    top_prob: float = Field(..., description="Probability of the model's preferred residue.")


class ScoreResult(BaseModel):
    """Output of the scoring adapter and ``POST /score`` body.

    ``perplexity = exp(-mean_log_prob)`` (lower = the sequence fits the structure
    better; uniform over 20 AAs ≈ 20). ``recovery`` is the fraction of positions
    where the model's top prediction equals the scored residue.
    """

    positions: list[ScorePosition]
    mean_log_prob: float
    perplexity: float
    recovery: float = Field(..., description="Fraction where model top-prediction == scored AA, in [0,1].")
    n_residues: int
    mode: str
    use_sequence: bool
    num_batches: int
    sequence_scored: str = Field(..., description="The amino-acid sequence that was scored.")
    inference_ms: float


class ScoreResponse(ScoreResult):
    """``POST /score`` response: the score result plus a request id."""

    request_id: str = Field(..., description="Unique id for this request (also in X-Request-ID).")


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
