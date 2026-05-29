"""FastAPI serving layer for UMA-Inverse.

Exposes the inference adapter as a REST service with response-time discipline,
Prometheus metrics, and structured logging. The Gradio UI (:mod:`src.serving.ui`)
is mounted at ``/`` when available.

Response-time guarantees (honest version)
-----------------------------------------
* The **residue cap** (``UMA_MAX_RESIDUES``) is the real protection: oversized
  structures are rejected with 413 *before* any decode begins.
* The request **timeout** (``UMA_REQUEST_TIMEOUT_S``) is a *response-level
  backstop only*. It returns 504 to the client, but a running inference thread
  cannot be cancelled in Python — the orphaned decode finishes on its own. With
  the cap set so normal requests complete under the timeout, this is rare.
* A **semaphore** (``UMA_MAX_CONCURRENCY``, default 1) serialises inference so
  concurrent requests queue instead of thrashing the (2 vCPU) box — which also
  keeps the latency percentiles meaningful.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from src.serving import metrics as M
from src.serving.inference import (
    DEFAULT_MAX_RESIDUES,
    InputTooLargeError,
    get_engine,
    run_inference,
)
from src.serving.metrics import configure_logging, get_logger
from src.serving.schemas import DesignRequest, DesignResponse, HealthResponse

# ── Runtime configuration (all env-overridable) ────────────────────────────────

REQUEST_TIMEOUT_S = float(os.environ.get("UMA_REQUEST_TIMEOUT_S", "300"))
MAX_CONCURRENCY = int(os.environ.get("UMA_MAX_CONCURRENCY", "1"))
EXAMPLES_DIR = Path(__file__).parent / "examples"

# Endpoints we track individually in metrics. Everything else (the Gradio UI's
# many hashed static assets, internal gradio_api calls) is bucketed under one
# label to keep the `endpoint` cardinality bounded — otherwise a single UI load
# spawns ~80 one-off counter series.
_TRACKED_ENDPOINTS = frozenset(
    {"/design", "/health", "/metrics", "/docs", "/openapi.json", "/"}
)


def _metric_endpoint(path: str) -> str:
    return path if path in _TRACKED_ENDPOINTS else "<ui>"

_START_TIME = time.time()
_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)


# ── Lifespan: load the model once, prime caches ─────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.environ.get("UMA_LOG_LEVEL", "INFO"))
    log = get_logger()

    # Optionally pin CPU threads to approximate the HF Spaces CPU Basic box.
    threads = os.environ.get("UMA_TORCH_THREADS")
    if threads:
        import torch

        torch.set_num_threads(int(threads))

    log.info("loading_model", event_detail="loading checkpoint on CPU")
    engine = await asyncio.get_running_loop().run_in_executor(None, get_engine)
    M.MODEL_LOAD_SECONDS.set(engine.model_load_seconds)
    log.info("model_loaded", model_load_seconds=round(engine.model_load_seconds, 2))

    # Warm-up on the smallest bundled example so the first real request isn't cold.
    warmup = _smallest_example()
    if warmup is not None:
        await asyncio.get_running_loop().run_in_executor(None, engine.warm_up, warmup)

    yield


def _smallest_example() -> Path | None:
    """Smallest ``.pdb`` under ``examples/`` (for warm-up), or None."""
    if not EXAMPLES_DIR.is_dir():
        return None
    pdbs = sorted(EXAMPLES_DIR.glob("*.pdb"), key=lambda p: p.stat().st_size)
    return pdbs[0] if pdbs else None


# ── App factory ─────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="UMA-Inverse Serving API",
        version="0.1.0",
        description=(
            "Ligand-conditioned protein inverse-folding served on CPU. "
            "POST a PDB structure to /design to get designed sequences with "
            "per-residue confidence. CPU demo — small structures only "
            f"(≤ {DEFAULT_MAX_RESIDUES} residues)."
        ),
        lifespan=lifespan,
    )

    _register_middleware(app)
    _register_exception_handlers(app)
    _register_routes(app)
    _mount_ui(app)
    return app


# ── Middleware: timing, metrics, structured logging, headers ────────────────────


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def observability(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.inference_ms = None
        request.state.input_residues = None
        request.state.mean_confidence = None
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        duration_ms = (time.perf_counter() - start) * 1000.0

        endpoint = request.url.path
        M.REQUESTS_TOTAL.labels(
            endpoint=_metric_endpoint(endpoint), status=str(response.status_code)
        ).inc()

        inf_ms = request.state.inference_ms
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Inference-MS"] = (
            f"{inf_ms:.2f}" if inf_ms is not None else f"{duration_ms:.2f}"
        )

        get_logger().info(
            "request",
            endpoint=endpoint,
            method=request.method,
            status=response.status_code,
            latency_ms=round(duration_ms, 2),
            inference_ms=inf_ms,
            input_residues=request.state.input_residues,
            mean_confidence=request.state.mean_confidence,
        )
        return response


# ── Exception handlers: structured JSON, never a stack trace ────────────────────


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InputTooLargeError)
    async def _too_large(request: Request, exc: InputTooLargeError):
        return JSONResponse(
            status_code=413,
            content={
                "error": "input_too_large",
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": "request failed schema validation",
                "errors": exc.errors(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(ValueError)
    async def _bad_value(request: Request, exc: ValueError):
        # e.g. a PDB with no parseable protein residues.
        return JSONResponse(
            status_code=400,
            content={
                "error": "bad_request",
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        get_logger().error("unhandled_error", error=str(exc), error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "inference failed; see server logs",
                "request_id": getattr(request.state, "request_id", None),
            },
        )


# ── Routes ───────────────────────────────────────────────────────────────────


def _register_routes(app: FastAPI) -> None:
    @app.post("/design", response_model=DesignResponse)
    async def design(req: DesignRequest, request: Request):
        loop = asyncio.get_running_loop()
        async with _SEMAPHORE:
            M.INFLIGHT_REQUESTS.inc()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: run_inference(
                            req.pdb,
                            ligand=req.ligand,
                            temperature=req.temperature,
                            n_samples=req.n_samples,
                        ),
                    ),
                    timeout=REQUEST_TIMEOUT_S,
                )
            except TimeoutError:
                # Backstop only: the orphaned thread keeps running to completion.
                return JSONResponse(
                    status_code=504,
                    content={
                        "error": "timeout",
                        "detail": (
                            f"inference exceeded {REQUEST_TIMEOUT_S:.0f}s. Try a "
                            "smaller structure or fewer samples."
                        ),
                        "request_id": request.state.request_id,
                    },
                )
            finally:
                M.INFLIGHT_REQUESTS.dec()

        # Record observability signals from the real result.
        M.record_design_metrics(
            n_residues=result.n_residues,
            mean_confidence=result.mean_confidence,
            inference_ms=result.inference_ms,
        )
        request.state.inference_ms = result.inference_ms
        request.state.input_residues = result.n_residues
        request.state.mean_confidence = result.mean_confidence

        return DesignResponse(request_id=request.state.request_id, **result.model_dump())

    @app.get("/health", response_model=HealthResponse)
    async def health():
        # Don't trigger a load here — report whether one has happened.
        from src.serving import inference as _inf

        loaded = _inf._engine is not None
        return HealthResponse(
            status="ok" if loaded else "starting",
            model_loaded=loaded,
            uptime_s=round(time.time() - _START_TIME, 1),
        )

    @app.get("/metrics")
    async def metrics():
        body, content_type = M.render_metrics()
        return Response(content=body, media_type=content_type)


# ── Gradio UI mount (optional) ──────────────────────────────────────────────────


def _mount_ui(app: FastAPI) -> None:
    """Mount the Gradio dashboard at ``/`` if it imports; else a JSON info route."""
    try:
        import gradio as gr

        from src.serving.ui import build_ui

        demo = build_ui()
        gr.mount_gradio_app(app, demo, path="/")
    except Exception as exc:  # noqa: BLE001 — UI is optional; API must still serve
        get_logger().warning("ui_mount_skipped", error=str(exc))

        @app.get("/")
        async def root():
            return {
                "service": "UMA-Inverse Serving API",
                "docs": "/docs",
                "endpoints": ["/design", "/health", "/metrics"],
            }


# Module-level ASGI app for ``uvicorn src.serving.app:app``.
app = create_app()
