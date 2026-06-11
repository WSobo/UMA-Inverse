"""Batch preprocess PDB files → cached ``.pt`` tensors (v5 cache).

Same union-cache schema as ``scripts/preprocess.py``; the v5 delta is in the
*data*, not in new tensor keys:

  * DNA/RNA ATOM records — routed into the ligand atom pool by the parser on
    the v5 branch (unconditional, not flag-gated). The v4 cache dropped these;
    keeping them is the whole point of rebuilding. Paired with
    ``ligand_context_atoms=50`` (the default here) so extended nucleic-acid
    pockets are not silently truncated.
  * ``--ligand_atoms 50`` (vs 25 in v4) — wider ligand-context budget for
    nucleic acids and large cofactors (NAD/FAD/heme).

Rich per-atom chemistry and covalent bond topology are intentionally DISABLED
— dropped from v5 scope (fewer changes at once; SOTA inverse-folding models
don't use them; and the RDKit pass is the slow/fragile bottleneck). Re-enabling
is an experiment decision, not just a flag flip.

Writes to ``data/processed_v5/`` by default so the v4 ``data/processed/`` cache
is untouched. CPU-only; safe to run alongside GPU training/benchmark jobs
(independent on-disk caches; only shares the filesystem).

Usage:
    uv run python scripts/preprocess_v5.py
    uv run python scripts/preprocess_v5.py --workers 32
    uv run python scripts/preprocess_v5.py --out_dir data/processed_v5
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
    """Featurise one structure with v5 keys included and save to disk.

    Returns (pdb_id, ok, reason). On RDKit failure for the rich features
    or bond topology, the bridge falls through to zero-filled tensors and
    we still write the cache entry — the model treats those as "no
    information" (consistent with the data layer's fallback).
    """
    out_path = os.path.join(out_dir, f"{pdb_id}.pt")
    if os.path.exists(out_path) and not recompute:
        return pdb_id, True, "cached"

    path = resolve_pdb_path(pdb_dir, pdb_id)
    if path is None:
        return pdb_id, False, "pdb_not_found"

    try:
        # Union cache (v5 schema). All v3 keys retained so datamodule
        # selections still work; v5 keys (ligand_rich_features +
        # ligand_bond_types) added when the RDKit fallback chain produces
        # them (or zero-filled tensors as a documented fallback).
        example = load_example_from_pdb(
            pdb_path=path,
            ligand_context_atoms=ligand_context_atoms,
            cutoff_for_score=cutoff_for_score,
            max_total_nodes=max_total_nodes,
            ligand_featurizer="atomic_number_embedding",
            residue_anchor="ca",
            return_backbone_coords=True,
            return_frame_relative_angles=True,
            return_sidechain_atoms=True,
            # Chemistry/bond-topology intentionally OFF — v5 scope keeps ONLY
            # the DNA/RNA routing (parser-level, automatic) + M=50 context.
            # Rich RDKit features were dropped from v5; do not re-enable here
            # without re-deciding the experiment (they are also the slow/
            # fragile RDKit-CPU bottleneck for a 155k-structure pass).
        )
        example["ligand_features"] = _encode_ligand_elements(
            example["ligand_atomic_numbers"]
        ).float()
        torch.save(example, out_path)
        return pdb_id, True, "processed"
    except (ValueError, OSError, RuntimeError) as e:
        return pdb_id, False, f"{type(e).__name__}:{e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PDBs for UMA-Inverse v5")
    parser.add_argument("--json_dir",   default=None,
                        help="Directory with train.json, valid.json. "
                             "Defaults to LigandMPNN/training/.")
    parser.add_argument("--pdb_dir",    default=None,
                        help="Raw PDB archive. Defaults to data/raw/pdb_archive/.")
    parser.add_argument("--out_dir",    default=None,
                        help="Output directory. Defaults to data/processed_v5/.")
    parser.add_argument("--max_nodes",  type=int,   default=1024,
                        help="max_total_nodes for featurisation (default: 1024).")
    parser.add_argument("--ligand_atoms", type=int, default=50,
                        help="ligand_context_atoms (default: 50 for v5 vs 25 in v4 "
                             "to accommodate DNA/RNA and larger cofactors).")
    parser.add_argument("--cutoff",     type=float, default=8.0,
                        help="cutoff_for_score in Å (default: 8.0).")
    parser.add_argument("--workers",    type=int,   default=None,
                        help="Worker processes. Defaults to os.cpu_count().")
    parser.add_argument("--recompute",  action="store_true",
                        help="Recompute even if cached .pt file already exists.")
    args = parser.parse_args()

    json_dir = args.json_dir or os.path.join(PROJECT_ROOT, "LigandMPNN", "training")
    pdb_dir  = args.pdb_dir  or os.path.join(PROJECT_ROOT, "data", "raw", "pdb_archive")
    out_dir  = args.out_dir  or os.path.join(PROJECT_ROOT, "data", "processed_v5")
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
    logger.info("Preprocessing %d structures → %s (v5 schema)", len(pdb_list), out_dir)

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
        "Done (v5). new=%d  cached=%d  failed=%d / total=%d",
        success, cached, failed, len(pdb_list),
    )

    if failed_list:
        fail_log = os.path.join(PROJECT_ROOT, "logs", "preprocess_v5_failures.txt")
        os.makedirs(os.path.dirname(fail_log), exist_ok=True)
        with open(fail_log, "w") as f:
            f.write("\n".join(failed_list) + "\n")
        logger.warning("Failed IDs written to %s", fail_log)


if __name__ == "__main__":
    main()
