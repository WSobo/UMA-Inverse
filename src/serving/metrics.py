"""Observability: Prometheus metrics + structured (JSON) logging.

All metrics are updated from *real* requests by the middleware in
:mod:`src.serving.app` — nothing here is hardcoded or simulated. The
histograms are what give p50/p90/p99 latency and the confidence distribution.

Buckets are sized for **CPU** autoregressive inference, which is slow and
length-dependent (seconds to tens of seconds), so the default Prometheus
buckets (capped at 10 s) would be useless here.
"""
from __future__ import annotations

import logging
import sys

import structlog
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ── Metric definitions (registered on the default global registry) ─────────────

REQUESTS_TOTAL = Counter(
    "uma_requests_total",
    "Total HTTP requests handled, by endpoint and status class.",
    ["endpoint", "status"],
)

INFERENCE_LATENCY = Histogram(
    "uma_inference_latency_seconds",
    "Wall-clock time spent decoding a design request (CPU).",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)

REQUEST_SIZE_RESIDUES = Histogram(
    "uma_request_size_residues",
    "Number of residues in the parsed input structure.",
    buckets=(10, 25, 50, 75, 100, 120, 150, 200, 300, 400),
)

MEAN_CONFIDENCE = Histogram(
    "uma_mean_confidence",
    "Aggregate per-request mean confidence (model prediction quality).",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

SCORE_PERPLEXITY = Histogram(
    "uma_score_perplexity",
    "Per-request sequence perplexity from /score (lower = sequence fits structure better).",
    buckets=(1.5, 2, 3, 4, 5, 7, 10, 14, 20),
)

INFLIGHT_REQUESTS = Gauge(
    "uma_inflight_requests",
    "Inference requests currently executing (design or score).",
)

MODEL_LOAD_SECONDS = Gauge(
    "uma_model_load_seconds",
    "Time taken to load the checkpoint at startup (set once).",
)


def record_design_metrics(*, n_residues: int, mean_confidence: float, inference_ms: float) -> None:
    """Record the observability signals for one completed design.

    Shared by the REST endpoint and the Gradio UI so both surfaces feed the
    same histograms — the live-metrics dashboard then reflects *all* real usage.
    """
    INFERENCE_LATENCY.observe(inference_ms / 1000.0)
    REQUEST_SIZE_RESIDUES.observe(n_residues)
    MEAN_CONFIDENCE.observe(mean_confidence)


def record_score_metrics(*, n_residues: int, perplexity: float, inference_ms: float) -> None:
    """Record observability signals for one completed /score request."""
    INFERENCE_LATENCY.observe(inference_ms / 1000.0)
    REQUEST_SIZE_RESIDUES.observe(n_residues)
    SCORE_PERPLEXITY.observe(perplexity)


def render_metrics() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the Prometheus exposition endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST


# ── Structured logging ─────────────────────────────────────────────────────────


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit one JSON line per event to stdout.

    Idempotent enough for app startup; safe to call once in the FastAPI
    lifespan. Routes stdlib logging through the same JSON pipeline so library
    logs (e.g. the inference session) are structured too.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging into structlog's JSON renderer.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)


def get_logger(name: str = "uma.serving") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
