"""Download PDB files listed in LigandMPNN training/validation/test JSON splits.

Usage:
    uv run python scripts/download_json_pdbs.py
    uv run python scripts/download_json_pdbs.py --limit 100
    uv run python scripts/download_json_pdbs.py --json_dir /path/to/custom/splits/

The JSON files are expected to contain a list of PDB IDs (or a dict with PDB
IDs as keys) — the standard LigandMPNN split format.
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_pdb(pdb_id: str, out_dir: str) -> bool:
    """Download a single PDB file from RCSB with up to 3 retries.

    Stores files in RCSB-style subdirectories: ``<out_dir>/<mid2>/<pdb_id>.pdb``
    (e.g. ``1abc`` → ``<out_dir>/ab/1abc.pdb``).
    """
    pdb_id = str(pdb_id).lower()
    sub_dir = pdb_id[1:3] if len(pdb_id) >= 4 else "misc"
    target_dir = os.path.join(out_dir, sub_dir)
    os.makedirs(target_dir, exist_ok=True)

    out_path = os.path.join(target_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        return True

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, out_path)
            return True
        except Exception:
            time.sleep(2 * (attempt + 1))

    logger.warning("Failed to fetch %s after 3 attempts", pdb_id)
    return False


def _load_ids_from_json(json_path: str) -> set:
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(str(x) for x in data)
    elif isinstance(data, dict):
        return set(data.keys())
    logger.warning("Unexpected JSON format in %s — expected list or dict", json_path)
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PDB files listed in LigandMPNN JSON splits."
    )
    parser.add_argument(
        "--json_dir",
        default=None,
        help=(
            "Directory containing train.json, valid.json, test_small_molecule.json. "
            "Defaults to LigandMPNN/training/ relative to the project root."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for PDB files. Defaults to data/raw/pdb_archive/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of structures fetched (useful for pilot runs).",
    )
    parser.add_argument(
        "--json-files",
        nargs="+",
        default=None,
        help=(
            "Explicit split filenames to read from --json_dir (overrides the default "
            "train.json / valid.json / test_small_molecule.json set). Example: "
            "--json-files test_metal.json test_nucleotide.json test_small_molecule.json"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download threads (default: 8).",
    )
    parser.add_argument(
        "--split-dirs",
        action="store_true",
        help=(
            "Place each JSON's PDBs in its own subdirectory named after the JSON "
            "file stem (e.g. test_metal.json → <out_dir>/test_metal/<xx>/<id>.pdb). "
            "Duplicated IDs across splits get downloaded once per subdir, but "
            "downstream tooling can treat each subdir as a self-contained archive."
        ),
    )
    args = parser.parse_args()

    json_dir = args.json_dir or os.path.join(PROJECT_ROOT, "LigandMPNN", "training")
    out_dir  = args.out_dir  or os.path.join(PROJECT_ROOT, "data", "raw", "pdb_archive")
    os.makedirs(out_dir, exist_ok=True)

    split_names = args.json_files or [
        "train.json",
        "valid.json",
        "test_small_molecule.json",
    ]

    # Build per-split (out_dir, id_list) groups. In merged mode (default) every
    # split deposits into the same out_dir after set-deduping; in split-dirs
    # mode each JSON writes into <out_dir>/<json_stem>/.
    groups: list[tuple[str, list[str]]] = []
    if args.split_dirs:
        for name in split_names:
            json_path = os.path.join(json_dir, name)
            if not os.path.exists(json_path):
                logger.debug("Split file not found (skipping): %s", json_path)
                continue
            ids = _load_ids_from_json(json_path)
            logger.info("Parsed %d IDs from %s", len(ids), json_path)
            stem = os.path.splitext(os.path.basename(name))[0]
            group_dir = os.path.join(out_dir, stem)
            os.makedirs(group_dir, exist_ok=True)
            groups.append((group_dir, list(ids)))
    else:
        merged_ids: set = set()
        for name in split_names:
            json_path = os.path.join(json_dir, name)
            if os.path.exists(json_path):
                ids = _load_ids_from_json(json_path)
                logger.info("Parsed %d IDs from %s", len(ids), json_path)
                merged_ids.update(ids)
            else:
                logger.debug("Split file not found (skipping): %s", json_path)
        groups.append((out_dir, list(merged_ids)))

    if not any(ids for _, ids in groups):
        logger.error(
            "No PDB IDs found. Check --json_dir (tried: %s). "
            "LigandMPNN must be cloned at %s for default paths.",
            json_dir,
            os.path.join(PROJECT_ROOT, "LigandMPNN"),
        )
        sys.exit(1)

    # Flatten to (id, dest_dir) pairs, respecting --limit across all groups.
    tasks: list[tuple[str, str]] = []
    for group_dir, ids in groups:
        for pid in ids:
            tasks.append((pid, group_dir))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    if args.split_dirs:
        per_group_counts = ", ".join(
            f"{os.path.basename(g)}={len(ids)}" for g, ids in groups
        )
        logger.info(
            "Downloading %d PDB files across %d split dirs → %s  (%s)",
            len(tasks), len(groups), out_dir, per_group_counts,
        )
    else:
        logger.info("Downloading %d unique PDB structures → %s", len(tasks), out_dir)

    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_pdb, pid, dest): (pid, dest) for pid, dest in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                success += 1
            if i % 500 == 0:
                logger.info("Progress: %d/%d (%d succeeded)", i, len(tasks), success)

    logger.info("Done. Prepared %d/%d PDB files in %s", success, len(tasks), out_dir)


if __name__ == "__main__":
    main()
