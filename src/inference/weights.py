"""Resolve the UMA-Inverse checkpoint path.

Used by the CLI (and ``scripts/download_weights.py``) to either honour an
explicit user-supplied checkpoint or auto-fetch the canonical weights
from Hugging Face Hub on first use.

Cache location (XDG-compliant), keyed by the checkpoint filename so a new
canonical checkpoint never silently reuses a stale cache from an older model:
    $XDG_CACHE_HOME/uma-inverse/<filename>
    ~/.cache/uma-inverse/<filename>   (default)
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical weights: the v5 model (epoch 11, val-loss 1.2175) that matches this
# repo's model code. Override per-deployment with UMA_CKPT_REPO / UMA_CKPT_FILE
# (e.g. a Hugging Face Space), or pin a local file with --ckpt / UMA_CKPT.
DEFAULT_REPO_ID = "WSobo/UMA-Inverse"
DEFAULT_FILENAME = "uma-inverse-11-1.2175.ckpt"

# Anything smaller is almost certainly a partial / failed download.
MIN_REASONABLE_CKPT_BYTES = 10 * 1024 * 1024  # 10 MB


def default_cache_path(filename: str = DEFAULT_FILENAME) -> Path:
    """Per-user cache path for a checkpoint, keyed by its filename.

    Keying on the filename means a new canonical checkpoint (different name)
    never silently reuses a stale cache from a previous model version.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "uma-inverse" / filename


def _looks_intact(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return path.stat().st_size >= MIN_REASONABLE_CKPT_BYTES


def fetch_default_checkpoint(
    *,
    output: Path | None = None,
    repo_id: str | None = None,
    filename: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    force: bool = False,
) -> Path:
    """Download the canonical checkpoint to ``output`` (default: user cache).

    ``repo_id`` / ``filename`` / ``revision`` fall back to the ``UMA_CKPT_REPO``
    / ``UMA_CKPT_FILE`` / ``UMA_CKPT_REVISION`` environment variables, then to
    the module defaults — so a deployment (e.g. a Hugging Face Space) can
    repoint the weights without code changes.

    Idempotent: if a usable file already lives at ``output``, returns it
    without re-downloading unless ``force=True``.
    """
    repo_id = repo_id or os.environ.get("UMA_CKPT_REPO") or DEFAULT_REPO_ID
    filename = filename or os.environ.get("UMA_CKPT_FILE") or DEFAULT_FILENAME
    revision = revision or os.environ.get("UMA_CKPT_REVISION")
    output = Path(output) if output is not None else default_cache_path(filename)

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
    - Otherwise, return (and lazily fetch) the canonical checkpoint
      from Hugging Face into the user cache.
    """
    if ckpt is not None:
        ckpt = Path(ckpt)
        if not ckpt.exists() or not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt
    return fetch_default_checkpoint()
