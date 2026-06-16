"""Paired interface-recovery stats: UMA-Inverse vs matched-protocol LigandMPNN.

For each ligand class we pair per-PDB median interface recovery on the PDB ids
scored by *both* models, then report a Wilcoxon signed-rank test and bootstrap
95% CIs (10k resamples, seed 0) on the paired difference and on each model's
mean-of-per-PDB-medians.

Source:
    outputs/benchmark/interface_recovery/v5-test_<split>/per_pdb.csv          (UMA-Inverse)
    outputs/benchmark/interface_recovery/ligandmpnn-test_<split>/per_pdb.csv  (LigandMPNN, matched)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "outputs" / "benchmark" / "interface_recovery"
SPLITS = ["small_molecule", "metal", "nucleotide"]
RNG = np.random.default_rng(0)
NBOOT = 10_000


def load(path: Path) -> dict[str, float]:
    d: dict[str, float] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                v = float(r["median_recovery"])
            except (KeyError, ValueError):
                continue
            if v == v:  # skip NaN
                d[r["pdb_id"]] = v
    return d


def boot_mean_ci(x: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), size=(NBOOT, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


for split in SPLITS:
    uma = load(BENCH / f"v5-test_{split}" / "per_pdb.csv")
    lmp = load(BENCH / f"ligandmpnn-test_{split}" / "per_pdb.csv")
    shared = sorted(set(uma) & set(lmp))
    u = np.array([uma[p] for p in shared])
    l = np.array([lmp[p] for p in shared])
    diff = u - l
    w = wilcoxon(u, l, alternative="two-sided")
    du_lo, du_hi = boot_mean_ci(diff)
    u_lo, u_hi = boot_mean_ci(u)
    l_lo, l_hi = boot_mean_ci(l)
    print(f"\n[{split}]  N_paired={len(shared)}  (UMA-only={len(uma)}, LMPNN-only={len(lmp)})")
    print(f"  UMA-Inverse  mean={u.mean()*100:.1f}%  95% CI [{u_lo*100:.1f}, {u_hi*100:.1f}]")
    print(f"  LigandMPNN   mean={l.mean()*100:.1f}%  95% CI [{l_lo*100:.1f}, {l_hi*100:.1f}]")
    print(f"  paired diff (UMA - LMPNN)  mean={diff.mean()*100:+.1f} pp  "
          f"95% CI [{du_lo*100:+.1f}, {du_hi*100:+.1f}]")
    print(f"  Wilcoxon  W={w.statistic:.1f}  p={w.pvalue:.2e}  "
          f"(UMA better on {int((diff>0).sum())}/{len(shared)} PDBs)")
