"""UMA-Inverse model-serving layer.

A thin, CPU-only serving wrapper around the existing inference library
(:mod:`src.inference`). Provides:

* :mod:`src.serving.inference` — a request-friendly ``run_inference`` adapter
  backed by a lazily-loaded singleton model.
* :mod:`src.serving.schemas` — Pydantic V2 request/response contracts.
* :mod:`src.serving.app` — the FastAPI application (REST + metrics + UI mount).

Nothing here retrains or reimplements the model: it imports and calls the
same code path the CLI (``uma-inverse design``) uses.
"""
