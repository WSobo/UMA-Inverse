"""Build Boltz-2 YAML inputs for the pocket-fixed cofold experiment.

For each PDB x method x sample (where method in {uma_v2, ligandmpnn} and we
sample 5 of the 20 generated sequences with a fixed seed), build a Boltz-2
input YAML at:

    outputs/preprint/boltz_inputs/cofold/<method>/<pdb_id>_sample<NN>.yaml

Then a single Boltz-2 invocation per method directory cofolds all 100 inputs
in one model-load (much faster than 100 separate invocations).

Output layout when complete:
    outputs/preprint/boltz_inputs/cofold/uma_v2/         100 YAMLs
    outputs/preprint/boltz_inputs/cofold/ligandmpnn/     100 YAMLs

Inputs:
    outputs/preprint/pdb_selection.json
    outputs/preprint/uma_pocket_fixed/<pdb>/designs.fasta
    outputs/preprint/ligandmpnn_pocket_fixed/seqs/<pdb>.fa

Each YAML follows the existing scripts/preprint/build_boltz_yaml.py template:
protein chain A + ligand (CCD code) + affinity property enabled.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprint.build_boltz_yaml import build_yaml  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_cofold_yamls")

# Standard amino acid codes — used to detect protein chains (vs DNA/RNA in 1qum).
_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "MSE",
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


def _protein_chain_lengths(pdb_path: Path) -> list[tuple[str, int]]:
    """Return [(chain_id, n_residues), ...] for protein chains in PDB iteration order.

    Iteration order follows BioPython's chain order (matches the order that UMA's
    pdb_parser and LigandMPNN concatenate when building the design sequence).
    Chains with zero standard-AA residues are dropped (e.g. DNA chains in 1qum).
    """
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    model = next(iter(structure))
    chains: list[tuple[str, int]] = []
    for chain in model:
        n = sum(
            1 for r in chain
            if r.get_id()[0] == " " and r.get_resname().strip().upper() in _STANDARD_AA
        )
        if n > 0:
            chains.append((chain.get_id(), n))
    return chains


def _split_design_into_chains(
    design_seq: str, chain_lengths: list[tuple[str, int]]
) -> list[tuple[str, str]] | None:
    """Split a flat design sequence at chain boundaries.

    Returns None if the lengths don't match (caller should warn + skip).
    """
    expected = sum(n for _cid, n in chain_lengths)
    if len(design_seq) != expected:
        return None
    chunks: list[tuple[str, str]] = []
    offset = 0
    for cid, n in chain_lengths:
        chunks.append((cid, design_seq[offset:offset + n]))
        offset += n
    return chunks


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    seq_lines: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines).replace(":", "")))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines).replace(":", "")))
    return records


def _load_uma_designs(pdb_id: str, uma_dir: Path) -> list[str]:
    fasta = uma_dir / pdb_id / "designs.fasta"
    if not fasta.exists():
        return []
    return [seq for _h, seq in _read_fasta(fasta)]


def _load_ligandmpnn_designs(pdb_id: str, lig_dir: Path) -> list[str]:
    """LigandMPNN's seqs/<pdb>.fa: first record is native, rest are designs."""
    fasta = lig_dir / "seqs" / f"{pdb_id}.fa"
    if not fasta.exists():
        return []
    records = _read_fasta(fasta)
    if len(records) < 2:
        return []
    return [seq for _h, seq in records[1:]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pdb_selection.json",
    )
    parser.add_argument(
        "--uma-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "uma_pocket_fixed",
    )
    parser.add_argument(
        "--ligandmpnn-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "ligandmpnn_pocket_fixed",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "boltz_inputs" / "cofold",
    )
    parser.add_argument("--samples-per-pdb", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "uma_v2").mkdir(exist_ok=True)
    (args.out_dir / "ligandmpnn").mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    selection = json.loads(args.selection.read_text())

    pdbs: list[tuple[str, str, dict]] = []
    for entry in selection["small_molecule"]:
        pdbs.append((entry["pdb_id"], "small_molecule", entry))
    for entry in selection["metal"]:
        pdbs.append((entry["pdb_id"], "metal", entry))

    n_emitted = {"uma_v2": 0, "ligandmpnn": 0}
    n_missing = {"uma_v2": 0, "ligandmpnn": 0}

    sampling_record: list[dict] = []  # remember which sample indices we picked

    for pdb_id, kind, entry in pdbs:
        ligand_value = entry.get("ccd_code") or entry.get("ion")
        if ligand_value is None:
            logger.warning("%s: no ligand identifier in selection JSON, skipping", pdb_id)
            continue

        # Resolve protein chain lengths for this PDB (for multi-chain split).
        pdb_dir = args.smallmol_pdb_dir if kind == "small_molecule" else args.metal_pdb_dir
        pdb_path = _resolve_pdb_path(pdb_dir, pdb_id)
        if pdb_path is None:
            logger.warning("%s: PDB file not found, skipping", pdb_id)
            continue
        chain_lengths = _protein_chain_lengths(pdb_path)
        if len(chain_lengths) > 1:
            logger.info("%s: multimer (%s)", pdb_id,
                         ", ".join(f"{cid}={n}" for cid, n in chain_lengths))

        def _emit(method: str, designs: list[str]) -> None:
            if not designs:
                logger.info("%s designs missing for %s -- will skip in cofold batch",
                             method, pdb_id)
                n_missing[method] += 1
                return
            sample_indices = rng.sample(range(len(designs)),
                                         min(args.samples_per_pdb, len(designs)))
            for s_idx in sample_indices:
                seq = designs[s_idx]
                if len(chain_lengths) == 1:
                    yaml_seq: str | list[tuple[str, str]] = seq
                else:
                    chunks = _split_design_into_chains(seq, chain_lengths)
                    if chunks is None:
                        logger.warning(
                            "%s sample %d: design length %d does not match chain "
                            "lengths %s -- emitting as single chain (may yield wrong "
                            "structure)",
                            pdb_id, s_idx, len(seq),
                            ", ".join(f"{c}={n}" for c, n in chain_lengths),
                        )
                        yaml_seq = seq
                    else:
                        yaml_seq = chunks
                # Pick a ligand chain id that doesn't collide with any protein chain.
                used_ids = {cid for cid, _ in chain_lengths}
                ligand_id = next(c for c in "LBCDEFGHIJKMNOPQRSTUVWXYZ" if c not in used_ids)
                yaml_text = build_yaml(
                    sequence=yaml_seq,
                    ligand_kind="ccd",
                    ligand_value=ligand_value,
                    ligand_id=ligand_id,
                    affinity=True,
                )
                out_path = args.out_dir / method / f"{pdb_id}_sample{s_idx:02d}.yaml"
                out_path.write_text(yaml_text)
                n_emitted[method] += 1
                sampling_record.append({
                    "pdb_id": pdb_id, "kind": kind, "method": method,
                    "sample_idx": s_idx, "yaml": str(out_path),
                })

        _emit("uma_v2", _load_uma_designs(pdb_id, args.uma_dir))
        _emit("ligandmpnn", _load_ligandmpnn_designs(pdb_id, args.ligandmpnn_dir))

    # Persist the sampling record for downstream metric computation
    record_path = args.out_dir / "sampling_record.json"
    record_path.write_text(json.dumps(sampling_record, indent=2))
    logger.info("wrote %s   (%d entries)", record_path, len(sampling_record))

    print(f"\nUMA YAMLs:        {n_emitted['uma_v2']:>3d}   (missing for {n_missing['uma_v2']} PDBs)")
    print(f"LigandMPNN YAMLs: {n_emitted['ligandmpnn']:>3d}   (missing for {n_missing['ligandmpnn']} PDBs)")
    print(f"Total cofolds to run: {sum(n_emitted.values())}")
    print(f"\nLaunch cofolds with:")
    print(f"  sbatch scripts/SLURM/preprint_boltz_cofold.sh")


if __name__ == "__main__":
    main()
