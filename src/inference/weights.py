"""Resolve the UMA-Inverse checkpoint path.

Used by the CLI (and ``scripts/download_weights.py``) to either honour an
explicit user-supplied checkpoint or auto-fetch the canonical v2 weights
from Hugging Face Hub on first use.

Cache location (XDG-compliant):
    $XDG_CACHE_HOME/uma-inverse/uma-inverse-v2.ckpt
    ~/.cache/uma-inverse/uma-inverse-v2.ckpt   (default)
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "WSobo/UMA-Inverse_v2"
DEFAULT_FILENAME = "uma-inverse-19-1.1463.ckpt"
DEFAULT_CACHE_NAME = "uma-inverse-v2.ckpt"

# Anything smaller is almost certainly a partial / failed download.
MIN_REASONABLE_CKPT_BYTES = 10 * 1024 * 1024  # 10 MB


def default_cache_path() -> Path:
    """Stable per-user cache path for the canonical checkpoint."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "uma-inverse" / DEFAULT_CACHE_NAME


def _looks_intact(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return path.stat().st_size >= MIN_REASONABLE_CKPT_BYTES


def fetch_default_checkpoint(
    *,
    output: Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    revision: str | None = None,
    token: str | None = None,
    force: bool = False,
) -> Path:
    """Download the canonical checkpoint to ``output`` (default: user cache).

    Idempotent: if a usable file already lives at ``output``, returns it
    without re-downloading unless ``force=True``.
    """
    output = Path(output) if output is not None else default_cache_path()

    if not force and _looks_intact(output):
        logger.info("checkpoint already present at %s — skipping download", output)
        return output

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch weights. "
            "Run `uv sync` (or `pip install huggingface_hub`)."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetching %s/%s from Hugging Face", repo_id, filename)
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
        )
    )
    if not _looks_intact(cached):
        size = cached.stat().st_size if cached.exists() else -1
        raise RuntimeError(
            f"downloaded file at {cached} does not look intact (size {size} bytes)"
        )
    if cached.resolve() != output.resolve():
        shutil.copy2(cached, output)
    logger.info("✓ checkpoint cached at %s (%.1f MB)",
                output, output.stat().st_size / (1024 * 1024))
    return output


def resolve_checkpoint(ckpt: Path | None) -> Path:
    """Return a usable checkpoint path for inference.

    - If ``ckpt`` is given, validate that the file exists and return it.
    - Otherwise, return (and lazily fetch) the canonical v2 checkpoint
      from Hugging Face into the user cache.
    """
    if ckpt is not None:
        ckpt = Path(ckpt)
        if not ckpt.exists() or not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt
    return fetch_default_checkpoint()
