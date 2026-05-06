"""Unified metric computation for pocket-fixed redesigns from both UMA and LigandMPNN.

For each PDB x method x sample, computes:
    - distal_recovery   : fraction of non-fixed positions where pred == native
    - pocket_recovery   : fraction of fixed positions where pred == native
                          (sanity: should be 1.0 -- if not, something's wrong
                          with the fix-residue plumbing)

For each (PDB, method) -- aggregating over the K samples:
    - mean / median / std of distal_recovery
    - mean pairwise Hamming distance at distal positions (sequence diversity)
    - per-AA frequency at distal positions (for the diversity / PSSM analysis)

Outputs:
    outputs/preprint/pocket_fixed_metrics.csv      -- one row per sample
    outputs/preprint/pocket_fixed_summary.csv      -- one row per (pdb, method)
    outputs/preprint/pocket_fixed_aa_freq.csv      -- one row per (pdb, method, aa)

Inputs:
    outputs/preprint/pdb_selection.json
    outputs/preprint/uma_pocket_fixed/<pdb_id>/designs.fasta
    outputs/preprint/ligandmpnn_pocket_fixed/seqs/<pdb_id>.fa
    data/raw/pdb_archive/test_{small_molecule,metal}/<pdb_id>.pdb
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pocket_fixed_metrics")

AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


def _resolve_pdb_path(pdb_dir: Path, pdb_id: str) -> Path | None:
    pid = pdb_id.lower()
    nested = pdb_dir / pid[1:3] / f"{pid}.pdb"
    if nested.exists():
        return nested
    flat = pdb_dir / f"{pid}.pdb"
    if flat.exists():
        return flat
    return None


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    """Return list of (header, sequence) records, sequences with `:` chain
    separators stripped."""
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


def _extract_native_seq_with_residue_ids(pdb_path: Path) -> tuple[str, list[str]]:
    """Return (native_seq_1letter, residue_ids) for chain A."""
    from Bio.PDB import PDBParser
    AA3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "HID": "H", "HIE": "H", "HIP": "H", "MSE": "M",
    }
    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    model = next(iter(structure))
    seq_parts: list[str] = []
    rids: list[str] = []
    for chain in model:
        chain_id = chain.get_id()
        for res in chain:
            het_flag, resnum, icode = res.get_id()
            if het_flag != " ":
                continue
            resname = res.get_resname().strip().upper()
            seq_parts.append(AA3_TO_1.get(resname, "X"))
            ic = icode.strip()
            rids.append(f"{chain_id}{resnum}{ic}" if ic else f"{chain_id}{resnum}")
    return "".join(seq_parts), rids


def _per_position_recovery(pred: str, native: str, mask: np.ndarray) -> float:
    """Fraction of positions where pred == native, restricted to True positions in mask."""
    if mask.sum() == 0:
        return float("nan")
    pred_arr = np.frombuffer(pred.encode(), dtype="S1")
    native_arr = np.frombuffer(native.encode(), dtype="S1")
    if pred_arr.shape != native_arr.shape:
        # truncate to min length (LigandMPNN sometimes adds chain separators)
        m = min(pred_arr.shape[0], native_arr.shape[0], mask.shape[0])
        pred_arr = pred_arr[:m]
        native_arr = native_arr[:m]
        mask = mask[:m]
    correct = (pred_arr == native_arr) & mask
    return float(correct.sum() / mask.sum())


def _pairwise_hamming(seqs: list[str], mask: np.ndarray) -> float:
    """Mean pairwise Hamming distance over the masked positions.

    Robust to length mismatches: trims `mask` to the shortest sequence
    (LigandMPNN sometimes drops 1 residue with a missing CA atom).
    """
    if not seqs or mask.sum() == 0:
        return float("nan")
    arrs = [np.frombuffer(s.encode(), dtype="S1") for s in seqs]
    m = min(mask.shape[0], min(a.shape[0] for a in arrs))
    mask_trim = mask[:m]
    if mask_trim.sum() == 0:
        return float("nan")
    seq_arrays = [a[:m][mask_trim] for a in arrs]
    K = len(seq_arrays)
    if K < 2:
        return 0.0
    n = int(mask_trim.sum())
    pairs = list(itertools.combinations(range(K), 2))
    total = 0.0
    for i, j in pairs:
        total += (seq_arrays[i] != seq_arrays[j]).sum() / n
    return total / len(pairs)


def _per_aa_freq(seqs: list[str], mask: np.ndarray) -> dict[str, float]:
    """Per-AA frequency at masked positions across all K sequences."""
    if not seqs or mask.sum() == 0:
        return {aa: 0.0 for aa in AA_LETTERS}
    counts: dict[str, int] = {aa: 0 for aa in AA_LETTERS}
    total = 0
    for s in seqs:
        a = np.frombuffer(s.encode(), dtype="S1")
        if a.shape[0] < mask.shape[0]:
            continue
        a_masked = a[: mask.shape[0]][mask]
        for ch in a_masked:
            chs = ch.decode()
            if chs in counts:
                counts[chs] += 1
                total += 1
    if total == 0:
        return {aa: 0.0 for aa in AA_LETTERS}
    return {aa: counts[aa] / total for aa in AA_LETTERS}


def _load_uma_designs(pdb_id: str, uma_dir: Path) -> list[str]:
    """Returns list of K design sequences (chain A) for a given PDB. Empty list if missing."""
    fasta = uma_dir / pdb_id / "designs.fasta"
    if not fasta.exists():
        return []
    records = _read_fasta(fasta)
    return [seq for _h, seq in records]


def _load_ligandmpnn_designs(pdb_id: str, lig_dir: Path) -> tuple[str | None, list[str]]:
    """Returns (native_seq_or_None, list_of_K_designs).

    LigandMPNN's FASTA: first record is native, rest are designs.
    """
    fasta = lig_dir / "seqs" / f"{pdb_id}.fa"
    if not fasta.exists():
        return None, []
    records = _read_fasta(fasta)
    if len(records) < 2:
        return None, []
    native = records[0][1]
    designs = [seq for _h, seq in records[1:]]
    return native, designs


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
        default=PROJECT_ROOT / "outputs" / "preprint",
    )
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    pdbs: list[tuple[str, str, dict]] = []
    for entry in selection["small_molecule"]:
        pdbs.append((entry["pdb_id"], "small_molecule", entry))
    for entry in selection["metal"]:
        pdbs.append((entry["pdb_id"], "metal", entry))

    per_sample_rows: list[dict] = []
    summary_rows: list[dict] = []
    aa_freq_rows: list[dict] = []

    for pdb_id, kind, entry in pdbs:
        pdb_dir = args.smallmol_pdb_dir if kind == "small_molecule" else args.metal_pdb_dir
        pdb_path = _resolve_pdb_path(pdb_dir, pdb_id)
        if pdb_path is None:
            logger.warning("PDB not found: %s — skipping", pdb_id)
            continue
        native_seq, residue_ids = _extract_native_seq_with_residue_ids(pdb_path)
        L = len(native_seq)

        # Build pocket / distal masks aligned to chain A residue order.
        pocket_set = set(entry["pocket_residues"])
        pocket_mask = np.array([rid in pocket_set for rid in residue_ids], dtype=bool)
        distal_mask = ~pocket_mask
        # Exclude X tokens from "designable" comparisons (LigandMPNN paper convention).
        valid_mask = np.array([aa != "X" for aa in native_seq], dtype=bool)
        distal_mask &= valid_mask
        pocket_mask &= valid_mask

        # ── UMA designs ─────────────────────────────────────────────────────
        uma_designs = _load_uma_designs(pdb_id, args.uma_dir)
        if uma_designs:
            for s_idx, seq in enumerate(uma_designs):
                per_sample_rows.append({
                    "pdb_id": pdb_id, "kind": kind, "method": "uma_v2",
                    "sample_idx": s_idx,
                    "distal_recovery": _per_position_recovery(seq, native_seq, distal_mask),
                    "pocket_recovery": _per_position_recovery(seq, native_seq, pocket_mask),
                })
            distals = [r["distal_recovery"] for r in per_sample_rows
                       if r["pdb_id"] == pdb_id and r["method"] == "uma_v2"]
            pocket_recs = [r["pocket_recovery"] for r in per_sample_rows
                           if r["pdb_id"] == pdb_id and r["method"] == "uma_v2"]
            summary_rows.append({
                "pdb_id": pdb_id, "kind": kind, "method": "uma_v2",
                "n_samples": len(uma_designs),
                "n_residues": L, "n_pocket": int(pocket_mask.sum()),
                "n_distal": int(distal_mask.sum()),
                "mean_distal_recovery": float(np.mean(distals)),
                "median_distal_recovery": float(np.median(distals)),
                "stdev_distal_recovery": float(np.std(distals, ddof=0)),
                "mean_pocket_recovery": float(np.mean(pocket_recs)),
                "mean_pairwise_hamming_distal": _pairwise_hamming(uma_designs, distal_mask),
            })
            for aa, freq in _per_aa_freq(uma_designs, distal_mask).items():
                aa_freq_rows.append({
                    "pdb_id": pdb_id, "kind": kind, "method": "uma_v2",
                    "aa": aa, "freq_distal": freq,
                })
        else:
            logger.info("UMA designs missing for %s", pdb_id)

        # ── LigandMPNN designs ──────────────────────────────────────────────
        lig_native, lig_designs = _load_ligandmpnn_designs(pdb_id, args.ligandmpnn_dir)
        if lig_designs:
            # Sanity: LigandMPNN's first record is the native -- verify it matches.
            if lig_native and lig_native != native_seq:
                logger.warning(
                    "%s: LigandMPNN native record differs from PDB-extracted native "
                    "(len %d vs %d) -- may indicate a chain ordering mismatch",
                    pdb_id, len(lig_native), len(native_seq),
                )
            for s_idx, seq in enumerate(lig_designs):
                per_sample_rows.append({
                    "pdb_id": pdb_id, "kind": kind, "method": "ligandmpnn",
                    "sample_idx": s_idx,
                    "distal_recovery": _per_position_recovery(seq, native_seq, distal_mask),
                    "pocket_recovery": _per_position_recovery(seq, native_seq, pocket_mask),
                })
            distals = [r["distal_recovery"] for r in per_sample_rows
                       if r["pdb_id"] == pdb_id and r["method"] == "ligandmpnn"]
            pocket_recs = [r["pocket_recovery"] for r in per_sample_rows
                           if r["pdb_id"] == pdb_id and r["method"] == "ligandmpnn"]
            summary_rows.append({
                "pdb_id": pdb_id, "kind": kind, "method": "ligandmpnn",
                "n_samples": len(lig_designs),
                "n_residues": L, "n_pocket": int(pocket_mask.sum()),
                "n_distal": int(distal_mask.sum()),
                "mean_distal_recovery": float(np.mean(distals)),
                "median_distal_recovery": float(np.median(distals)),
                "stdev_distal_recovery": float(np.std(distals, ddof=0)),
                "mean_pocket_recovery": float(np.mean(pocket_recs)),
                "mean_pairwise_hamming_distal": _pairwise_hamming(lig_designs, distal_mask),
            })
            for aa, freq in _per_aa_freq(lig_designs, distal_mask).items():
                aa_freq_rows.append({
                    "pdb_id": pdb_id, "kind": kind, "method": "ligandmpnn",
                    "aa": aa, "freq_distal": freq,
                })
        else:
            logger.info("LigandMPNN designs missing for %s", pdb_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if per_sample_rows:
        path = args.out_dir / "pocket_fixed_metrics.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            for r in per_sample_rows:
                writer.writerow(r)
        logger.info("wrote %s   (%d rows)", path, len(per_sample_rows))

    if summary_rows:
        path = args.out_dir / "pocket_fixed_summary.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        logger.info("wrote %s   (%d rows)", path, len(summary_rows))

    if aa_freq_rows:
        path = args.out_dir / "pocket_fixed_aa_freq.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aa_freq_rows[0].keys()))
            writer.writeheader()
            for r in aa_freq_rows:
                writer.writerow(r)
        logger.info("wrote %s   (%d rows)", path, len(aa_freq_rows))

    print("\n--- Summary by method ---")
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in summary_rows:
        by_method[r["method"]].append(r)
    for method, rows in by_method.items():
        if not rows:
            continue
        mean_distal = float(np.mean([r["mean_distal_recovery"] for r in rows]))
        mean_hamming = float(np.mean([r["mean_pairwise_hamming_distal"] for r in rows]))
        mean_pocket = float(np.mean([r["mean_pocket_recovery"] for r in rows]))
        print(f"  {method:12s} n_pdbs={len(rows):>2d}  distal_recovery={mean_distal:.3f}  "
              f"hamming_diversity={mean_hamming:.3f}  pocket_recovery={mean_pocket:.3f}")


if __name__ == "__main__":
    main()
