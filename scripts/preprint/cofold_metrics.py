"""Phase B2: parse Boltz-2 cofold outputs and compute pocket-fixed comparison metrics.

For each (pdb_id, method, sample) input recorded in `sampling_record.json`,
locate the corresponding Boltz-2 output:

    outputs/preprint/cofold/<method>/boltz_results_<method>/predictions/<basename>/
        <basename>_model_N.cif|pdb         (5 diffusion samples, N=0..4)
        confidence_<basename>_model_N.json (per-sample confidences)
        affinity_<basename>.json           (one per input)

Compute:
    - Boltz-2 confidence (best-of-5, mean-of-5): confidence_score, ptm, iptm,
      ligand_iptm, complex_plddt, complex_iplddt
    - Boltz-2 affinity head: pred_value, probability_binary (one per input)
    - Pocket Calpha RMSD: between native crystal pocket residues and the
      Boltz-2 cofold's chain-A pocket residues, after Kabsch alignment on
      pocket Calphas. (Restricted to the residues we forced fixed in A2/A3.)
    - Ligand-pose RMSD: native ligand heavy atoms vs Boltz-2 predicted ligand
      heavy atoms, after the pocket-Calpha alignment above.
    - Whole-protein scaffold RMSD: chain-A Calpha RMSD between native and
      Boltz-2 cofold (the design's overall conformational drift).

Output: outputs/preprint/cofold_metrics.csv (one row per cofold input,
selecting the model_0 confidence + best-iptm-model RMSDs).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cofold_metrics")


def _resolve_pdb_path(pdb_dir: Path, pdb_id: str) -> Path | None:
    pid = pdb_id.lower()
    nested = pdb_dir / pid[1:3] / f"{pid}.pdb"
    if nested.exists():
        return nested
    flat = pdb_dir / f"{pid}.pdb"
    if flat.exists():
        return flat
    return None


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimal rigid alignment of P onto Q via Kabsch.

    Returns (R, t, rmsd) where R is the rotation matrix, t the translation,
    and rmsd the post-alignment Calpha RMSD.
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    P_aligned = (R @ P.T).T + t
    rmsd = float(np.sqrt(((P_aligned - Q) ** 2).sum(axis=1).mean()))
    return R, t, rmsd


def _native_pocket_and_ligand(
    pdb_path: Path, pocket_residue_ids: set[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Returns ({rid: native_Calpha_xyz}, native_ligand_atoms_xyz)."""
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    model = next(iter(structure))

    native_pocket: dict[str, np.ndarray] = {}
    native_ligand: list[list[float]] = []

    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            het_flag, resnum, icode = residue.get_id()
            ic = icode.strip()
            rid = f"{chain_id}{resnum}{ic}" if ic else f"{chain_id}{resnum}"
            if het_flag == " ":
                if rid in pocket_residue_ids and "CA" in residue:
                    native_pocket[rid] = np.array(residue["CA"].get_coord(), dtype=np.float64)
            elif het_flag != "W":
                resname = residue.get_resname().strip().upper()
                if resname in {"HOH", "WAT"}:
                    continue
                for atom in residue.get_atoms():
                    elem = (atom.element or atom.get_name()[0]).strip().upper()
                    if elem in {"H", "D"}:
                        continue
                    native_ligand.append(list(atom.get_coord()))
    return native_pocket, np.array(native_ligand, dtype=np.float64) if native_ligand else np.zeros((0, 3))


def _cofold_pocket_and_ligand(
    cif_path: Path, pocket_residue_ids: set[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Same as _native_pocket_and_ligand but for a Boltz-2 CIF (chain A protein + chain B ligand)."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_path))
    model = next(iter(structure))

    cofold_pocket: dict[str, np.ndarray] = {}
    cofold_ligand: list[list[float]] = []

    # Boltz-2 outputs renumber the chain-A residues 1..L. We need to map back
    # to the native pocket residue IDs by *position in chain A*. The caller
    # is responsible for producing pocket_residue_ids in chain-A residue-number
    # form (e.g. "A23"), and we'll match by the integer suffix here.
    pocket_nums: dict[int, str] = {}
    for rid in pocket_residue_ids:
        if rid[0] != "A":
            continue
        try:
            num = int("".join(c for c in rid[1:] if c.isdigit()))
        except ValueError:
            continue
        pocket_nums[num] = rid

    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            het_flag, resnum, icode = residue.get_id()
            if chain_id == "A" and het_flag == " ":
                if resnum in pocket_nums and "CA" in residue:
                    cofold_pocket[pocket_nums[resnum]] = np.array(
                        residue["CA"].get_coord(), dtype=np.float64
                    )
            elif chain_id != "A":
                # Ligand chain. Boltz-2 names it "B" by default per our YAMLs.
                for atom in residue.get_atoms():
                    elem = (atom.element or atom.get_name()[0]).strip().upper()
                    if elem in {"H", "D"}:
                        continue
                    cofold_ligand.append(list(atom.get_coord()))

    return cofold_pocket, np.array(cofold_ligand, dtype=np.float64) if cofold_ligand else np.zeros((0, 3))


def _all_calphas(path: Path) -> np.ndarray:
    """Chain-A Calphas, ordered by residue number."""
    from Bio.PDB import PDBParser
    from Bio.PDB.MMCIFParser import MMCIFParser
    if path.suffix.lower() in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(path))
    model = next(iter(structure))
    if "A" not in [c.get_id() for c in model]:
        return np.zeros((0, 3))
    chain = model["A"]
    coords: list[list[float]] = []
    nums: list[int] = []
    for res in chain:
        het_flag, resnum, _ic = res.get_id()
        if het_flag != " ":
            continue
        if "CA" in res:
            coords.append(list(res["CA"].get_coord()))
            nums.append(resnum)
    order = np.argsort(nums)
    coords_arr = np.array(coords, dtype=np.float64)
    return coords_arr[order]


def _extract_metrics_for_input(
    base: str,
    method_dir: Path,
    pocket_residue_ids: set[str],
    native_pdb_path: Path,
) -> dict | None:
    """Parse all 5 model cofolds + confidences + affinity for one Boltz-2 input."""
    pred_dir = method_dir / f"boltz_results_{method_dir.name}" / "predictions" / base
    if not pred_dir.exists():
        return None

    confidences: list[dict] = []
    rmsds_per_model: list[dict] = []

    for model_idx in range(5):
        conf_json = pred_dir / f"confidence_{base}_model_{model_idx}.json"
        if not conf_json.exists():
            continue
        try:
            conf = json.loads(conf_json.read_text())
            confidences.append(conf)
        except Exception as exc:
            logger.warning("conf parse failed for %s model %d: %s", base, model_idx, exc)
            continue

        cif = pred_dir / f"{base}_model_{model_idx}.cif"
        pdb = pred_dir / f"{base}_model_{model_idx}.pdb"
        struct_path = cif if cif.exists() else pdb
        if not struct_path.exists():
            continue

        try:
            native_pocket, native_lig = _native_pocket_and_ligand(native_pdb_path, pocket_residue_ids)
            cofold_pocket, cofold_lig = _cofold_pocket_and_ligand(struct_path, pocket_residue_ids)
            common = sorted(set(native_pocket) & set(cofold_pocket))
            if len(common) < 3:
                rmsds_per_model.append({"pocket_calpha_rmsd": float("nan"),
                                          "ligand_rmsd": float("nan"),
                                          "scaffold_rmsd": float("nan")})
                continue
            P = np.stack([cofold_pocket[r] for r in common])
            Q = np.stack([native_pocket[r] for r in common])
            R, t, pocket_rmsd = _kabsch(P, Q)

            # Apply same alignment to all cofold ligand atoms, compute RMSD to native ligand.
            if cofold_lig.size and native_lig.size and cofold_lig.shape[0] == native_lig.shape[0]:
                cofold_lig_aligned = (R @ cofold_lig.T).T + t
                ligand_rmsd = float(np.sqrt(((cofold_lig_aligned - native_lig) ** 2).sum(axis=1).mean()))
            else:
                ligand_rmsd = float("nan")

            # Whole-protein scaffold RMSD: align all chain-A Calphas to native chain-A Calphas.
            cofold_ca = _all_calphas(struct_path)
            native_ca = _all_calphas(native_pdb_path)
            n = min(cofold_ca.shape[0], native_ca.shape[0])
            if n >= 3:
                _, _, scaffold_rmsd = _kabsch(cofold_ca[:n], native_ca[:n])
            else:
                scaffold_rmsd = float("nan")

            rmsds_per_model.append({
                "pocket_calpha_rmsd": pocket_rmsd,
                "ligand_rmsd": ligand_rmsd,
                "scaffold_rmsd": scaffold_rmsd,
            })
        except Exception as exc:
            logger.warning("rmsd computation failed for %s model %d: %s", base, model_idx, exc)
            rmsds_per_model.append({"pocket_calpha_rmsd": float("nan"),
                                      "ligand_rmsd": float("nan"),
                                      "scaffold_rmsd": float("nan")})

    if not confidences:
        return None

    # Affinity (one per input)
    affinity_path = pred_dir / f"affinity_{base}.json"
    affinity: dict = {}
    if affinity_path.exists():
        try:
            affinity = json.loads(affinity_path.read_text())
        except Exception as exc:
            logger.warning("affinity parse failed for %s: %s", base, exc)

    def _agg(field: str, conf_list: list[dict], rmsd_list: list[dict]) -> tuple[float, float, float]:
        """Return (best, mean, stdev) for a field. Field name lookup:
           confidence keys: confidence_score, iptm, ligand_iptm, ptm, complex_plddt, complex_iplddt
           rmsd keys:       pocket_calpha_rmsd, ligand_rmsd, scaffold_rmsd"""
        if field in ("confidence_score", "iptm", "ligand_iptm", "ptm",
                      "complex_plddt", "complex_iplddt"):
            vals = [c.get(field) for c in conf_list if c.get(field) is not None]
        else:
            vals = [r.get(field) for r in rmsd_list if r.get(field) is not None and not np.isnan(r.get(field, np.nan))]
        if not vals:
            return float("nan"), float("nan"), float("nan")
        # for confidence, "best" = max; for rmsd, "best" = min
        best = max(vals) if field not in ("pocket_calpha_rmsd", "ligand_rmsd", "scaffold_rmsd") else min(vals)
        return float(best), float(statistics.fmean(vals)), float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0

    out: dict = {}
    for field in ("confidence_score", "iptm", "ligand_iptm", "ptm",
                  "complex_plddt", "complex_iplddt"):
        b, m, s = _agg(field, confidences, [])
        out[f"{field}_best"] = b
        out[f"{field}_mean"] = m
        out[f"{field}_stdev"] = s
    for field in ("pocket_calpha_rmsd", "ligand_rmsd", "scaffold_rmsd"):
        b, m, s = _agg(field, [], rmsds_per_model)
        out[f"{field}_best"] = b
        out[f"{field}_mean"] = m
        out[f"{field}_stdev"] = s

    out["affinity_pred_value"] = affinity.get("affinity_pred_value", float("nan"))
    out["affinity_probability_binary"] = affinity.get("affinity_probability_binary", float("nan"))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sampling-record",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "boltz_inputs" / "cofold" / "sampling_record.json",
    )
    parser.add_argument(
        "--cofold-base",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "cofold",
    )
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
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics.csv",
    )
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    by_pdb: dict[str, dict] = {}
    for entry in selection["small_molecule"]:
        by_pdb[entry["pdb_id"]] = {"kind": "small_molecule", **entry}
    for entry in selection["metal"]:
        by_pdb[entry["pdb_id"]] = {"kind": "metal", **entry}

    sampling = json.loads(args.sampling_record.read_text())

    rows: list[dict] = []
    for record in sampling:
        pdb_id = record["pdb_id"]
        method = record["method"]
        s_idx = record["sample_idx"]
        kind = record["kind"]
        sel = by_pdb[pdb_id]
        pocket_set = set(sel["pocket_residues"])
        pdb_dir = args.smallmol_pdb_dir if kind == "small_molecule" else args.metal_pdb_dir
        native_pdb_path = _resolve_pdb_path(pdb_dir, pdb_id)
        if native_pdb_path is None:
            logger.warning("native PDB not found for %s -- skipping", pdb_id)
            continue

        method_dir = args.cofold_base / method
        # The Boltz-2 input filename was <pdb_id>_sample<NN>.yaml so the
        # base used in the output dir tree is <pdb_id>_sample<NN>.
        base = f"{pdb_id}_sample{s_idx:02d}"

        metrics = _extract_metrics_for_input(base, method_dir, pocket_set, native_pdb_path)
        if metrics is None:
            logger.info("no cofold output found for %s [%s]", base, method)
            continue

        rows.append({
            "pdb_id": pdb_id,
            "kind": kind,
            "method": method,
            "sample_idx": s_idx,
            **metrics,
        })

    if not rows:
        raise SystemExit("no cofold outputs to summarize -- has Phase B1 finished?")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("wrote %s   (%d rows)", args.out, len(rows))


if __name__ == "__main__":
    main()
