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

        # UMA designs
        uma_designs = _load_uma_designs(pdb_id, args.uma_dir)
        if not uma_designs:
            logger.info("UMA designs missing for %s -- will skip in cofold batch", pdb_id)
            n_missing["uma_v2"] += 1
        else:
            sample_indices = rng.sample(range(len(uma_designs)),
                                         min(args.samples_per_pdb, len(uma_designs)))
            for s_idx in sample_indices:
                seq = uma_designs[s_idx]
                yaml_text = build_yaml(
                    sequence=seq,
                    ligand_kind="ccd",
                    ligand_value=ligand_value,
                    affinity=True,
                )
                out_path = args.out_dir / "uma_v2" / f"{pdb_id}_sample{s_idx:02d}.yaml"
                out_path.write_text(yaml_text)
                n_emitted["uma_v2"] += 1
                sampling_record.append({
                    "pdb_id": pdb_id, "kind": kind, "method": "uma_v2",
                    "sample_idx": s_idx, "yaml": str(out_path),
                })

        # LigandMPNN designs
        lig_designs = _load_ligandmpnn_designs(pdb_id, args.ligandmpnn_dir)
        if not lig_designs:
            logger.info("LigandMPNN designs missing for %s -- will skip in cofold batch", pdb_id)
            n_missing["ligandmpnn"] += 1
        else:
            sample_indices = rng.sample(range(len(lig_designs)),
                                         min(args.samples_per_pdb, len(lig_designs)))
            for s_idx in sample_indices:
                seq = lig_designs[s_idx]
                yaml_text = build_yaml(
                    sequence=seq,
                    ligand_kind="ccd",
                    ligand_value=ligand_value,
                    affinity=True,
                )
                out_path = args.out_dir / "ligandmpnn" / f"{pdb_id}_sample{s_idx:02d}.yaml"
                out_path.write_text(yaml_text)
                n_emitted["ligandmpnn"] += 1
                sampling_record.append({
                    "pdb_id": pdb_id, "kind": kind, "method": "ligandmpnn",
                    "sample_idx": s_idx, "yaml": str(out_path),
                })

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
