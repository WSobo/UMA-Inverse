"""Pre-fetch UMA-Inverse v2 weights from Hugging Face.

Most users do **not** need to run this — the ``uma-inverse`` CLI
auto-fetches the default checkpoint into ``~/.cache/uma-inverse/`` on its
first call. This script exists for situations where you want the download
to happen explicitly (e.g. on a machine without network access at
inference time, or to populate a project-local ``checkpoints/`` directory).

Usage:
    uv run python scripts/download_weights.py
    uv run python scripts/download_weights.py --force
    uv run python scripts/download_weights.py --output checkpoints/uma-inverse-v2.ckpt
    uv run python scripts/download_weights.py --repo-id user/repo --filename ckpt.ckpt

After download, you can pass ``--ckpt`` explicitly:

    uv run uma-inverse design --pdb my_complex.pdb --ckpt <output> ...
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.weights import (  # noqa: E402
    DEFAULT_FILENAME,
    DEFAULT_REPO_ID,
    default_cache_path,
    fetch_default_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repo (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--filename", default=DEFAULT_FILENAME,
        help=f"File to fetch from the repo (default: {DEFAULT_FILENAME}).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Local destination path (default: {default_cache_path()}).",
    )
    parser.add_argument(
        "--revision", default=None,
        help="Specific revision/branch/tag to fetch. Default: main.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the destination file already exists.",
    )
    parser.add_argument(
        "--token", default=None,
        help="HF auth token (only needed for private repos). "
             "Falls back to HF_TOKEN env var or huggingface-cli login.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = fetch_default_checkpoint(
            output=args.output,
            repo_id=args.repo_id,
            filename=args.filename,
            revision=args.revision,
            token=args.token,
            force=args.force,
        )
    except Exception as exc:
        logger.error("download failed: %s", exc)
        return 1

    logger.info("")
    logger.info("Next step: run inference (the CLI will reuse this cached file):")
    logger.info("  uv run uma-inverse design --pdb my_complex.pdb")
    logger.info("(or pass --ckpt %s explicitly)", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
