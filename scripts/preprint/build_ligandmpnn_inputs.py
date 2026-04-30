"""Build LigandMPNN/run.py inputs from outputs/preprint/pdb_selection.json.

LigandMPNN's run.py supports per-PDB fixed-residue lists via:

    --pdb_path_multi           json: {pdb_path: ""}   (only keys are used)
    --fixed_residues_multi     json: {pdb_path: "A12 A13 A14 B2 B25"}

We emit both files into outputs/preprint/ligandmpnn_inputs/ so the SLURM
wrapper can reference them by path.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_ligandmpnn_inputs")


def _resolve_pdb_path(pdb_dir: Path, pdb_id: str) -> Path | None:
    pid = pdb_id.lower()
    nested = pdb_dir / pid[1:3] / f"{pid}.pdb"
    if nested.exists():
        return nested
    flat = pdb_dir / f"{pid}.pdb"
    if flat.exists():
        return flat
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pdb_selection.json",
    )
    parser.add_argument(
        "--metal-pdb-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "pdb_archive" / "test_metal",
    )
    parser.add_argument(
        "--smallmol-pdb-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "pdb_archive" / "test_small_molecule",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "ligandmpnn_inputs",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(args.selection.read_text())

    pdb_path_multi: dict[str, str] = {}
    fixed_residues_multi: dict[str, str] = {}

    for kind, entries in (("small_molecule", selection["small_molecule"]),
                           ("metal", selection["metal"])):
        pdb_dir = args.smallmol_pdb_dir if kind == "small_molecule" else args.metal_pdb_dir
        for entry in entries:
            pdb_id = entry["pdb_id"]
            pdb_path = _resolve_pdb_path(pdb_dir, pdb_id)
            if pdb_path is None:
                logger.error("PDB not found: %s in %s", pdb_id, pdb_dir)
                continue
            abs_path = str(pdb_path.resolve())
            pdb_path_multi[abs_path] = ""
            # LigandMPNN expects space-separated A12 A13 ... format
            fixed_residues_multi[abs_path] = " ".join(entry["pocket_residues"])
            logger.info(
                "%s [%s]: %d pocket residues (sample: %s)",
                pdb_id, kind, len(entry["pocket_residues"]),
                ", ".join(entry["pocket_residues"][:3]),
            )

    pdb_path_json = args.out_dir / "pdb_path_multi.json"
    fixed_residues_json = args.out_dir / "fixed_residues_multi.json"

    pdb_path_json.write_text(json.dumps(pdb_path_multi, indent=2))
    fixed_residues_json.write_text(json.dumps(fixed_residues_multi, indent=2))

    logger.info("wrote %s   (%d PDBs)", pdb_path_json, len(pdb_path_multi))
    logger.info("wrote %s   (%d entries)", fixed_residues_json, len(fixed_residues_multi))

    print(f"\nLigandMPNN/run.py invocation snippet:")
    print(f"  python LigandMPNN/run.py \\")
    print(f"    --model_type ligand_mpnn \\")
    print(f"    --pdb_path_multi {pdb_path_json} \\")
    print(f"    --fixed_residues_multi {fixed_residues_json} \\")
    print(f"    --out_folder outputs/preprint/ligandmpnn_pocket_fixed \\")
    print(f"    --batch_size 4 --number_of_batches 5 \\")
    print(f"    --temperature 0.1 \\")
    print(f"    --seed 0 \\")
    print(f"    --save_stats 1")
    print(f"\n(K=20 sequences = batch_size 4 x number_of_batches 5)")


if __name__ == "__main__":
    main()
