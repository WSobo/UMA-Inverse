"""Select 15 ADDITIONAL small-molecule PDBs for the pocket-fixed scale-up.

Companion to ``scripts/preprint/select_pdbs.py``. Same hard filters, same
pocket-Jaccard dedup, but **drops the v2-friendly tiebreaker** so the
extended set isn't biased toward PDBs the model already recovers well.
This lets us run a sensitivity analysis: do findings #1 (small-mol distal
recovery edge) and #2 (low diversity / confident conservatism) survive a
selection regime that doesn't favor UMA?

Concretely the original selection picked, per CCD, the candidate with the
highest ``v2_mean_recovery``. Here we sort alphabetically by PDB id within
each CCD instead — deterministic, but unrelated to model performance.

Outputs:
    outputs/preprint/pdb_selection_extended.json   (15 new small_mol entries)
    outputs/preprint/pdb_selection_combined.json   (10 original + 15 new = 25
                                                     small_mol; metal split
                                                     unchanged at 10)

The combined JSON is what downstream metric / figure scripts read. The
extended JSON is what the new UMA / LigandMPNN / Boltz cofold runs read so
they only do work on the 15 new PDBs (the original 10 are already cofolded).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse parsing + filter helpers from the original selection script.
from scripts.preprint.select_pdbs import (  # type: ignore
    FUNCTIONAL_METALS,
    ARTIFACT_METALS,
    _apply_hard_filter,
    _ligand_ccd_codes,
    _parse_pdb,
    _pocket_residues,
    _read_per_pdb_csv,
    _resolve_pdb_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("select_extended_smallmol_pdbs")

# Excluded as ligand identity: crystallographic additives + buffers.
_NON_LIGAND_CCDS = (
    FUNCTIONAL_METALS | ARTIFACT_METALS
    | {"CL", "SO4", "PO4", "ACT", "EDO", "GOL", "PEG", "FMT",
       "DMS", "MES", "TRS", "EPE", "BME", "MPD", "IMD"}
)


def _build_candidates(rows: list[dict], pdb_dir: Path) -> list[dict]:
    """Apply per-PDB filters (pocket size, valid CCD) and return candidate dicts."""
    candidates: list[dict] = []
    for r in rows:
        path = _resolve_pdb_path(pdb_dir, r["pdb_id"])
        if path is None:
            continue
        try:
            parsed = _parse_pdb(path)
        except Exception:
            continue

        ccds = _ligand_ccd_codes(parsed)
        non_metal_ccds = ccds - _NON_LIGAND_CCDS
        if not non_metal_ccds:
            continue

        atoms_per_ccd: dict[str, int] = defaultdict(int)
        for ccd, _elem, _xyz in parsed["ligand_atoms"]:
            atoms_per_ccd[ccd] += 1
        primary_ccd = max(non_metal_ccds, key=lambda c: atoms_per_ccd.get(c, 0))

        pocket = _pocket_residues(parsed, cutoff=5.0)
        if not (5 <= len(pocket) <= 30):
            continue

        candidates.append(
            {
                "pdb_id": r["pdb_id"],
                "ccd_code": primary_ccd,
                "ligand_smiles": None,
                "pocket_residues": pocket,
                "n_residues": parsed["n_residues"],
                "n_pocket": len(pocket),
                "v2_mean_recovery": r["mean_recovery"],
                "selection_reason":
                    "extended set (no v2-recovery tiebreaker; "
                    "deterministic alphabetical pick per CCD)",
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-selection",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pdb_selection.json",
        help="Original 10+10 selection. Its small_mol PDBs and CCDs are excluded "
             "from the extended pool so the two sets are disjoint.",
    )
    parser.add_argument(
        "--per-pdb-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark" / "interface_recovery"
                / "v2-ep19-test_small_molecule" / "per_pdb.csv",
    )
    parser.add_argument(
        "--smallmol-pdb-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "pdb_archive" / "test_small_molecule",
    )
    parser.add_argument("--n-target", type=int, default=15,
                         help="Number of additional small-mol PDBs to pick.")
    parser.add_argument(
        "--out-extended",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pdb_selection_extended.json",
    )
    parser.add_argument(
        "--out-combined",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pdb_selection_combined.json",
    )
    args = parser.parse_args()

    existing = json.loads(args.existing_selection.read_text())
    excluded_pdb_ids = {e["pdb_id"] for e in existing["small_molecule"]}
    excluded_ccds = {e["ccd_code"] for e in existing["small_molecule"]}
    logger.info("excluding %d existing small_mol PDBs and %d CCDs",
                len(excluded_pdb_ids), len(excluded_ccds))

    rows = _read_per_pdb_csv(args.per_pdb_csv)
    rows = [r for r in rows if r["pdb_id"] not in excluded_pdb_ids]
    logger.info("starting pool: %d small_mol PDBs (after excluding existing)",
                len(rows))

    rows = _apply_hard_filter(rows)
    logger.info("after hard filter (size, pocket size): %d candidates", len(rows))

    candidates = _build_candidates(rows, args.smallmol_pdb_dir)
    logger.info("after per-PDB parse + pocket size + valid CCD: %d candidates",
                len(candidates))

    # Skip CCDs already in the original set
    candidates = [c for c in candidates if c["ccd_code"] not in excluded_ccds]
    logger.info("after CCD-disjoint filter: %d candidates", len(candidates))

    # Within each CCD, deterministic alphabetical pick (no v2-recovery preference).
    by_ccd: dict[str, dict] = {}
    for c in sorted(candidates, key=lambda c: (c["ccd_code"], c["pdb_id"])):
        by_ccd.setdefault(c["ccd_code"], c)
    deduped = sorted(by_ccd.values(), key=lambda c: c["pdb_id"])
    logger.info("after distinct-CCD dedup: %d candidates", len(deduped))

    # Same Jaccard pocket-overlap dedup as the original (avoids same-protein-different-inhibitor).
    selected: list[dict] = []
    for c in deduped:
        pocket_set = set(c["pocket_residues"])
        is_dup = False
        for s in selected:
            other_set = set(s["pocket_residues"])
            jaccard = len(pocket_set & other_set) / max(len(pocket_set | other_set), 1)
            if jaccard >= 0.6:
                is_dup = True
                logger.info(
                    "pocket-overlap dedup: dropping %s (CCD=%s) -- %d%% overlap with %s",
                    c["pdb_id"], c["ccd_code"], int(jaccard * 100), s["pdb_id"],
                )
                break
        if not is_dup:
            selected.append(c)
        if len(selected) >= args.n_target:
            break

    if len(selected) < args.n_target:
        logger.warning("only %d candidates available (target %d) -- using all",
                       len(selected), args.n_target)

    extended = {"small_molecule": selected, "metal": []}
    combined = {
        "small_molecule": existing["small_molecule"] + selected,
        "metal": existing["metal"],
    }

    args.out_extended.write_text(json.dumps(extended, indent=2))
    args.out_combined.write_text(json.dumps(combined, indent=2))
    logger.info("wrote %s  (%d new small_mol PDBs)",
                args.out_extended, len(selected))
    logger.info("wrote %s  (%d small_mol + %d metal = %d total)",
                args.out_combined,
                len(combined["small_molecule"]), len(combined["metal"]),
                len(combined["small_molecule"]) + len(combined["metal"]))

    print(f"\n=== Extended selection ({len(selected)} new small-mol PDBs) ===")
    for c in selected:
        print(f"  {c['pdb_id']:6s} CCD={c['ccd_code']:4s} L={c['n_residues']:3d} "
              f"pocket={c['n_pocket']:>2d} v2_recovery={c['v2_mean_recovery']:.3f}")


if __name__ == "__main__":
    main()
