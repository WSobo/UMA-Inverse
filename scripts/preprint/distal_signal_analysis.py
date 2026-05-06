"""Test the "intelligent distal redesign" claim: is UMA-Inverse's per-PDB
distal-posterior width a meaningful signal of design quality?

Three correlation tests, restricted to the small-molecule split (where the
diversity-gap finding lives):

  Test 1  Per-PDB Hamming-diversity correlation between UMA and LigandMPNN.
          If both architectures respond to the same signal (e.g. backbone),
          their per-PDB diversity rankings should agree. If UMA tracks
          something LigandMPNN cannot (ligand atoms via dense attention),
          they should be uncorrelated.

  Test 2  Per-PDB AA-distribution agreement at distal positions, between
          methods. High agreement + uncorrelated diversity (Test 1) ->
          'same target, different posterior width' picture.

  Test 4  Per-PDB confidence (1 - hamming) vs Boltz-2 cofold quality.
          Compute correlations with ipTM, pLDDT, pocket Calpha RMSD, ligand
          RMSD, predicted affinity. Run separately for each method. If UMA
          confidence is a true design-quality signal, UMA correlations should
          be strong/significant; LigMPNN correlations should be weak/null.

Test 3 (pocket-size dependence of diversity) is folded in as a side check.

Outputs:
    Prints a summary block to stdout. Designed to be re-run after the
    cofold-rerun (jobs 32977095/32977096) lands, which extends N from 21
    to 25 small-molecule PDBs.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _agg(rows: list[dict], field: str) -> float:
    vals = [
        float(r[field])
        for r in rows
        if r[field] not in ("", "nan") and float(r[field]) == float(r[field])
    ]
    return statistics.fmean(vals) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pocket_fixed_summary.csv",
    )
    parser.add_argument(
        "--aa-freq",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pocket_fixed_aa_freq.csv",
    )
    parser.add_argument(
        "--cofold",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics.csv",
    )
    args = parser.parse_args()

    pf = list(csv.DictReader(args.summary.open()))
    pf_by: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in pf:
        pf_by[r["pdb_id"]][r["method"]] = r

    cf = list(csv.DictReader(args.cofold.open()))
    cf_by: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in cf:
        cf_by[r["pdb_id"]][r["method"]].append(r)

    sm_pdbs_pf = [
        p for p, d in pf_by.items()
        if d.get("uma_v2", {}).get("kind") == "small_molecule" and "ligandmpnn" in d
    ]
    sm_pdbs_cf = [
        p for p in sm_pdbs_pf
        if "uma_v2" in cf_by[p] and "ligandmpnn" in cf_by[p]
    ]

    print(f"Small-mol paired (pocket-fixed):                  N = {len(sm_pdbs_pf)}")
    print(f"Small-mol paired (pocket-fixed AND cofold both):  N = {len(sm_pdbs_cf)}")

    # Test 1: per-PDB hamming correlation between methods
    uma_h = [float(pf_by[p]["uma_v2"]["mean_pairwise_hamming_distal"]) for p in sm_pdbs_pf]
    lig_h = [float(pf_by[p]["ligandmpnn"]["mean_pairwise_hamming_distal"]) for p in sm_pdbs_pf]
    print("\n[Test 1] Per-PDB distal-hamming correlation between UMA and LigandMPNN:")
    r, p = pearsonr(uma_h, lig_h); sr, sp = spearmanr(uma_h, lig_h)
    print(f"  Pearson  r = {r:+.3f}  p = {p:.3g}")
    print(f"  Spearman r = {sr:+.3f}  p = {sp:.3g}")
    print("  Reading: r near zero -> per-PDB diversity rankings are independent;")
    print("           UMA's confidence is set by a signal LigandMPNN doesn't track.")

    # Test 2: per-PDB AA-distribution correlation
    aa = list(csv.DictReader(args.aa_freq.open()))
    aa_by: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in aa:
        aa_by[r["pdb_id"]][r["method"]][r["aa"]] = float(r["freq_distal"])
    correls = []
    for p in sm_pdbs_pf:
        AAs = sorted(aa_by[p]["uma_v2"].keys())
        u = [aa_by[p]["uma_v2"][a] for a in AAs]
        l = [aa_by[p]["ligandmpnn"][a] for a in AAs]
        if sum(u) > 0 and sum(l) > 0:
            r, _ = pearsonr(u, l)
            correls.append(r)
    print("\n[Test 2] Per-PDB AA-distribution agreement between methods (Pearson r):")
    print(f"  median = {statistics.median(correls):.3f}, mean = {statistics.mean(correls):.3f}")
    print(f"  range = [{min(correls):.3f}, {max(correls):.3f}], N = {len(correls)}")
    print("  Reading: high agreement on which AAs to pick -> 'same target,")
    print("           different posterior width' picture.")

    # Test 3: pocket size vs diversity, per method
    sizes = [int(pf_by[p]["uma_v2"]["n_pocket"]) for p in sm_pdbs_pf]
    print("\n[Test 3] Pocket size vs distal hamming (Pearson r):")
    for label, h in (("UMA-v2     ", uma_h), ("LigandMPNN ", lig_h)):
        r, p = pearsonr(sizes, h)
        print(f"  {label}  r = {r:+.3f}  p = {p:.3g}")
    print("  Reading: pocket-residue count alone doesn't predict UMA confidence.")

    # Test 4: per-PDB confidence vs cofold quality
    print(f"\n[Test 4] Per-PDB distal confidence (1 - hamming) vs cofold quality, N = {len(sm_pdbs_cf)}")
    quality_fields = [
        ("affinity_probability_binary", +1, "higher = better"),
        ("affinity_pred_value",         -1, "lower = better"),
        ("iptm_best",                   +1, "higher = better"),
        ("complex_plddt_best",          +1, "higher = better"),
        ("pocket_calpha_rmsd_best",     -1, "lower = better"),
        ("ligand_rmsd_best",            -1, "lower = better"),
    ]
    for method in ("uma_v2", "ligandmpnn"):
        conf = [
            1.0 - float(pf_by[p][method]["mean_pairwise_hamming_distal"])
            for p in sm_pdbs_cf
        ]
        print(f"\n  --- {method} confidence vs {method} cofold quality ---")
        for field, sign, sense in quality_fields:
            quality = [sign * _agg(cf_by[p][method], field) for p in sm_pdbs_cf]
            valid = [(c, q) for c, q in zip(conf, quality) if not (c != c or q != q)]
            if len(valid) < 5:
                continue
            cs, qs = zip(*valid)
            r, p = pearsonr(cs, qs)
            sr, _sp = spearmanr(cs, qs)
            print(f"    {field:32s} ({sense:18s}): "
                  f"Pearson r = {r:+.3f} (p = {p:.3g}), Spearman r = {sr:+.3f}, N = {len(valid)}")

    print("\n" + "=" * 78)
    print("Story (if Test 4 holds):")
    print("  UMA confidence at distal positions tracks design quality.")
    print("  LigandMPNN confidence at distal positions does NOT.")
    print("  -> Dense pair attention transmits a real, design-quality-correlated")
    print("     signal to distal residues; KNN message-passing does not reach.")
    print("=" * 78)


if __name__ == "__main__":
    main()
