"""Batch preprocess PDB files → cached ``.pt`` tensors.

Reads PDB IDs from LigandMPNN split JSONs, featurises each structure, and
saves the result to ``data/processed/<pdb_id>.pt``.  Subsequent training runs
skip already-cached files unless ``--recompute`` is set.

Usage:
    uv run python scripts/preprocess.py
    uv run python scripts/preprocess.py --max_nodes 1024 --workers 32
"""
import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.ligandmpnn_bridge import (
    _encode_ligand_elements,
    load_example_from_pdb,
    load_json_ids,
    resolve_pdb_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _process_one(
    pdb_id: str,
    pdb_dir: str,
    out_dir: str,
    max_total_nodes: int,
    ligand_context_atoms: int,
    cutoff_for_score: float,
    recompute: bool,
) -> tuple[str, bool, str]:
    """Featurise one structure and save to disk. Returns (pdb_id, ok, reason)."""
    out_path = os.path.join(out_dir, f"{pdb_id}.pt")
    if os.path.exists(out_path) and not recompute:
        return pdb_id, True, "cached"

    path = resolve_pdb_path(pdb_dir, pdb_id)
    if path is None:
        return pdb_id, False, "pdb_not_found"

    try:
        # Union cache — emit every v2 key so a single cache serves any
        # combination of data.* flags without re-preprocessing. Anchor is
        # cached as Cα (the baseline); Dataset derives virtual Cβ on-the-fly
        # when data.residue_anchor="cb" is set. Both ligand_features (onehot6)
        # and ligand_atomic_numbers are written so the featurizer flag is a
        # pure selection at load time.
        example = load_example_from_pdb(
            pdb_path=path,
            ligand_context_atoms=ligand_context_atoms,
            cutoff_for_score=cutoff_for_score,
            max_total_nodes=max_total_nodes,
            ligand_featurizer="atomic_number_embedding",
            residue_anchor="ca",
            return_backbone_coords=True,
        )
        # Synthesize the v1 onehot6 features from the atomic numbers so the
        # cache can serve either featurizer selection. For zero-ligand
        # structures this returns the expected [0, 6] empty float tensor.
        example["ligand_features"] = _encode_ligand_elements(
            example["ligand_atomic_numbers"]
        ).float()
        torch.save(example, out_path)
        return pdb_id, True, "processed"
    except (ValueError, OSError, RuntimeError) as e:
        return pdb_id, False, f"{type(e).__name__}:{e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PDBs for UMA-Inverse")
    parser.add_argument("--json_dir",   default=None,
                        help="Directory with train.json, valid.json. "
                             "Defaults to LigandMPNN/training/.")
    parser.add_argument("--pdb_dir",    default=None,
                        help="Raw PDB archive. Defaults to data/raw/pdb_archive/.")
    parser.add_argument("--out_dir",    default=None,
                        help="Output directory. Defaults to data/processed/.")
    parser.add_argument("--max_nodes",  type=int,   default=1024,
                        help="max_total_nodes for featurisation (default: 1024).")
    parser.add_argument("--ligand_atoms", type=int, default=25,
                        help="ligand_context_atoms (default: 25).")
    parser.add_argument("--cutoff",     type=float, default=8.0,
                        help="cutoff_for_score in Å (default: 8.0).")
    parser.add_argument("--workers",    type=int,   default=None,
                        help="Worker processes. Defaults to os.cpu_count().")
    parser.add_argument("--recompute",  action="store_true",
                        help="Recompute even if cached .pt file already exists.")
    args = parser.parse_args()

    json_dir = args.json_dir or os.path.join(PROJECT_ROOT, "LigandMPNN", "training")
    pdb_dir  = args.pdb_dir  or os.path.join(PROJECT_ROOT, "data", "raw", "pdb_archive")
    out_dir  = args.out_dir  or os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    all_ids: set[str] = set()
    for name in ["train.json", "valid.json"]:
        json_path = os.path.join(json_dir, name)
        if os.path.exists(json_path):
            ids = load_json_ids(json_path)
            all_ids.update(ids)
            logger.info("Loaded %d IDs from %s", len(ids), json_path)
        else:
            logger.warning("Split file not found: %s", json_path)

    if not all_ids:
        logger.error("No IDs found. Exiting.")
        sys.exit(1)

    pdb_list = list(all_ids)
    logger.info("Preprocessing %d structures → %s", len(pdb_list), out_dir)

    workers = args.workers or os.cpu_count() or 4
    success = failed = cached = 0
    failed_list: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                pid, pdb_dir, out_dir,
                args.max_nodes, args.ligand_atoms, args.cutoff, args.recompute,
            ): pid
            for pid in pdb_list
        }
        for i, future in enumerate(as_completed(futures), 1):
            pid, ok, reason = future.result()
            if ok:
                if reason == "cached":
                    cached += 1
                else:
                    success += 1
            else:
                failed += 1
                failed_list.append(f"{pid}\t{reason}")
                logger.debug("FAIL %s: %s", pid, reason)
            if i % 1000 == 0:
                logger.info(
                    "Progress %d/%d — new:%d cached:%d failed:%d",
                    i, len(pdb_list), success, cached, failed,
                )

    logger.info(
        "Done. new=%d  cached=%d  failed=%d / total=%d",
        success, cached, failed, len(pdb_list),
    )

    if failed_list:
        fail_log = os.path.join(PROJECT_ROOT, "logs", "preprocess_failures.txt")
        os.makedirs(os.path.dirname(fail_log), exist_ok=True)
        with open(fail_log, "w") as f:
            f.write("\n".join(failed_list) + "\n")
        logger.warning("Failed IDs written to %s", fail_log)


if __name__ == "__main__":
    main()
