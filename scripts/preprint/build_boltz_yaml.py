"""Build Boltz-2 input YAML files for the pocket-fixed redesign experiment.

Inputs:
    - outputs/preprint/pdb_selection.json : the 20 PDBs from select_pdbs.py
    - the native protein sequence for each PDB (extracted from PDB ATOM records)
    - the ligand CCD code (already in the JSON)

For the SMOKE TEST mode (--smoke), build a single YAML for one selected PDB
using the *native* protein sequence. Used to verify the Boltz-2 pipeline runs
end-to-end before launching the full 200-cofold batch.

For the PRODUCTION mode (--mode batch), build N YAMLs from a FASTA of
designed sequences (one per (pdb_id, method, sample) tuple). Used by Phase B1.

YAML schema (per Boltz-2 docs):
    version: 1
    sequences:
        - protein:
            id: [A]
            sequence: <AAs>
            cyclic: false
        - ligand:
            id: [B]
            ccd: <3-letter CCD>
    properties:
        - affinity:
            binder: B
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_boltz_yaml")

# 3-letter to 1-letter amino acid map. X for unknowns.
AA_3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common variants
    "HID": "H", "HIE": "H", "HIP": "H",
    "MSE": "M",  # selenomethionine -> methionine
}


def _resolve_pdb_path(pdb_dir: Path, pdb_id: str) -> Path | None:
    pid = pdb_id.lower()
    nested = pdb_dir / pid[1:3] / f"{pid}.pdb"
    if nested.exists():
        return nested
    flat = pdb_dir / f"{pid}.pdb"
    if flat.exists():
        return flat
    return None


def extract_native_sequence(pdb_path: Path, chain_id: str = "A") -> str:
    """Return the AAs in chain `chain_id` as a 1-letter string, ordered by resnum."""
    try:
        from Bio.PDB import PDBParser
    except ImportError as exc:
        raise ImportError("biopython is required") from exc

    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    model = next(iter(structure))
    if chain_id not in [c.get_id() for c in model]:
        raise ValueError(f"chain {chain_id} not in {pdb_path}; available: "
                         f"{[c.get_id() for c in model]}")
    chain = model[chain_id]
    residues = []
    for res in chain:
        het_flag, _resnum, _icode = res.get_id()
        if het_flag != " ":
            continue  # skip HETATMs/waters
        resname = res.get_resname().strip().upper()
        residues.append(AA_3_TO_1.get(resname, "X"))
    return "".join(residues)


def build_yaml(
    sequence: str,
    *,
    ligand_kind: str,  # "ccd" or "smiles"
    ligand_value: str,
    affinity: bool = True,
) -> str:
    """Render a Boltz-2 input YAML. Returns the YAML text."""
    if ligand_kind not in ("ccd", "smiles"):
        raise ValueError(f"ligand_kind must be 'ccd' or 'smiles', got {ligand_kind!r}")

    # Indent the sequence to match YAML style (no folding — Boltz-2 should
    # tolerate a single long line).
    yaml = textwrap.dedent(f"""\
        version: 1
        sequences:
            - protein:
                id: [A]
                sequence: {sequence}
                cyclic: false
            - ligand:
                id: [B]
                {ligand_kind}: {"'" + ligand_value + "'" if ligand_kind == "smiles" else ligand_value}
        """)
    if affinity:
        yaml += textwrap.dedent("""\
            properties:
                - affinity:
                    binder: B
            """)
    return yaml


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
        default=PROJECT_ROOT / "outputs" / "preprint" / "boltz_inputs",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "native"],
        default="native",
        help="smoke: a single YAML for testing. native: one YAML per selected "
             "PDB using its native sequence (a sanity baseline before the design batch).",
    )
    parser.add_argument(
        "--smoke-pdb",
        type=str,
        default="2iwx",
        help="PDB id to build the smoke-test YAML for (default 2iwx, smallest small_mol PDB)",
    )
    parser.add_argument(
        "--no-affinity",
        action="store_true",
        help="Omit the affinity property (faster cofold, but no affinity score).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(args.selection.read_text())
    by_id = {}
    for entry in selection["small_molecule"]:
        by_id[entry["pdb_id"]] = ("small_molecule", entry)
    for entry in selection["metal"]:
        by_id[entry["pdb_id"]] = ("metal", entry)

    def _emit_yaml(pdb_id: str, kind_entry: tuple[str, dict]) -> Path:
        kind, entry = kind_entry
        pdb_dir = args.smallmol_pdb_dir if kind == "small_molecule" else args.metal_pdb_dir
        pdb_path = _resolve_pdb_path(pdb_dir, pdb_id)
        if pdb_path is None:
            raise FileNotFoundError(f"PDB {pdb_id} not found in {pdb_dir}")

        seq = extract_native_sequence(pdb_path, chain_id="A")
        if kind == "small_molecule":
            ligand_value = entry["ccd_code"]
        else:
            ligand_value = entry["ion"]

        yaml_text = build_yaml(
            sequence=seq,
            ligand_kind="ccd",
            ligand_value=ligand_value,
            affinity=not args.no_affinity,
        )
        out_path = args.out_dir / f"{pdb_id}_native.yaml"
        out_path.write_text(yaml_text)
        logger.info(
            "wrote %s  (chain A: %d residues, ligand %s/%s)",
            out_path, len(seq), kind, ligand_value,
        )
        return out_path

    if args.mode == "smoke":
        if args.smoke_pdb not in by_id:
            raise SystemExit(
                f"--smoke-pdb={args.smoke_pdb} not in selection. Available: {sorted(by_id)}"
            )
        out = _emit_yaml(args.smoke_pdb, by_id[args.smoke_pdb])
        print(f"\nSmoke test YAML ready: {out}")
        print("\nRun with the existing example template:")
        print(f"  sbatch scripts/SLURM/run_boltz_example.sh \\\n    {out} \\")
        print(f"    {PROJECT_ROOT}/outputs/preprint/boltz_smoke_results")
        return

    # mode == "native": build one YAML per selected PDB
    for pdb_id, kind_entry in by_id.items():
        try:
            _emit_yaml(pdb_id, kind_entry)
        except Exception as exc:
            logger.error("failed for %s: %s", pdb_id, exc)


if __name__ == "__main__":
    main()
