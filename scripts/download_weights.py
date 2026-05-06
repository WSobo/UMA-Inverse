"""Download UMA-Inverse v2 model weights from Hugging Face.

Usage:
    uv run python scripts/download_weights.py
    uv run python scripts/download_weights.py --force         # re-download even if present
    uv run python scripts/download_weights.py --repo-id user/repo --filename ckpt.ckpt

Default behaviour: fetches the canonical v2 checkpoint from the
configured Hugging Face repo into ``checkpoints/uma-inverse-v2.ckpt``
(relative to the project root). If the file is already there and looks
intact, the download is skipped — pass ``--force`` to override.

After a successful download, you can run inference:

    uv run uma-inverse design --pdb my_complex.pdb \\
        --ckpt checkpoints/uma-inverse-v2.ckpt --num-samples 10
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "WSobo/UMA-Inverse_v2"
DEFAULT_FILENAME = "uma-inverse-19-1.1463.ckpt"
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints" / "uma-inverse-v2.ckpt"

# A checkpoint smaller than this is almost certainly a partial/failed download.
MIN_REASONABLE_CKPT_BYTES = 10 * 1024 * 1024  # 10 MB


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
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Local destination path (default: {DEFAULT_OUTPUT}).",
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


def _looks_intact(path: Path) -> bool:
    """Conservative sanity check that a previously-downloaded ckpt is usable."""
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size < MIN_REASONABLE_CKPT_BYTES:
        logger.warning(
            "existing file %s is suspiciously small (%d bytes); will re-download",
            path, path.stat().st_size,
        )
        return False
    return True


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. Run `uv sync` to install dependencies."
        )
        return 1

    output: Path = args.output.resolve()

    if output.exists() and not args.force:
        if _looks_intact(output):
            logger.info("checkpoint already present at %s — skipping download", output)
            logger.info("(pass --force to re-download)")
            return 0

    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("fetching %s/%s", args.repo_id, args.filename)
    if args.revision:
        logger.info("revision: %s", args.revision)
    logger.info("destination: %s", output)

    try:
        cached_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            revision=args.revision,
            token=args.token,
        )
    except RepositoryNotFoundError:
        logger.error(
            "repository %s not found on Hugging Face. "
            "If the model is private, pass --token or set HF_TOKEN.",
            args.repo_id,
        )
        return 1
    except HfHubHTTPError as exc:
        logger.error("Hugging Face HTTP error: %s", exc)
        return 1
    except Exception as exc:
        logger.error("download failed: %s", exc)
        return 1

    cached_path = Path(cached_path)
    if not _looks_intact(cached_path):
        logger.error(
            "downloaded file at %s does not look intact (size %d bytes)",
            cached_path,
            cached_path.stat().st_size if cached_path.exists() else -1,
        )
        return 1

    # huggingface_hub returns a path inside its cache; copy into the project's
    # checkpoints/ tree so users have a stable, predictable location to reference.
    if cached_path.resolve() != output:
        shutil.copy2(cached_path, output)

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("✓ downloaded %s (%.1f MB)", output, size_mb)
    logger.info("")
    logger.info("Next step:")
    logger.info(
        "  uv run uma-inverse design --pdb my_complex.pdb \\\n"
        "      --ckpt %s --num-samples 10",
        output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
