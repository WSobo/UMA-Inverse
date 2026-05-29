# UMA-Inverse serving image — CPU-only, for Hugging Face Spaces (Docker SDK).
#
# The project pins the CUDA 12.4 torch wheel for GPU training (see
# [tool.uv.sources] in pyproject.toml). That wheel is huge and pointless on a
# CPU Space, so this image installs the CPU torch wheel FIRST; the subsequent
# project install then sees torch>=2.5 already satisfied and never pulls CUDA.
# Verify after a build: `pip list | grep -iE "torch|nvidia"` should show only
# a +cpu torch and no nvidia-* packages.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Run from the copied source tree (/app/src) so configs/ and the bundled
    # examples/ resolve relative to the code, not site-packages.
    PYTHONPATH=/app \
    UMA_CONFIG_PATH=/app/configs/config.yaml \
    GRADIO_ANALYTICS_ENABLED=False \
    # Caches (incl. the HF Hub checkpoint) go to a writable, ephemeral location.
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/tmp/.cache/huggingface \
    # Approximate HF Spaces CPU Basic (2 vCPU) and bound interactive latency.
    UMA_TORCH_THREADS=2 \
    UMA_MAX_RESIDUES=256 \
    UMA_MAX_CONCURRENCY=1 \
    UMA_REQUEST_TIMEOUT_S=300

# uv from the official distroless image (fast, reproducible installs).
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app

# 1) CPU torch FIRST — this is the whole point (no CUDA wheels).
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"

# 2) Project + serving deps. torch is already satisfied, so this resolves the
#    rest (fastapi, gradio, prometheus, structlog, biopython, lightning, …)
#    from PyPI without touching CUDA.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN uv pip install --system --no-cache ".[serving]"

EXPOSE 7860

# One worker (the model is a single CPU process); concurrency is bounded by the
# app-level semaphore plus uvicorn --limit-concurrency.
CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--limit-concurrency", "16"]
