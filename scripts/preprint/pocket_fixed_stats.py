"""Paired statistics on UMA vs LigandMPNN pocket-fixed redesign metrics.

For each metric of interest (mean distal recovery, mean pairwise Hamming
distance at distal positions), compute a paired Wilcoxon signed-rank test
across the PDBs (UMA paired with LigandMPNN on the same PDB).

Effect sizes:
    - Wilcoxon r = Z / sqrt(N) (also report W statistic)
    - Mean of paired differences (UMA - LigandMPNN)
    - 95% CI on the mean difference via bootstrapping (10,000 resamples)

Output: outputs/preprint/pocket_fixed_stats.txt (human-readable + parseable).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pocket_fixed_stats")

METRICS = (
    ("mean_distal_recovery", "distal recovery (mean over K samples)"),
    ("mean_pairwise_hamming_distal", "distal sequence diversity (mean pairwise Hamming)"),
    ("mean_pocket_recovery", "pocket recovery (sanity, should be 1.0 for both methods)"),
)


def _load_summary(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            for k, v in list(row.items()):
                if k in {"pdb_id", "kind", "method"}:
                    continue
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def _bootstrap_ci(diffs: np.ndarray, n_resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        means.append(float(sample.mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _wilcoxon(uma_vals: np.ndarray, lig_vals: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test with effect size estimates."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {
            "test": "wilcoxon",
            "error": "scipy not installed",
            "W": None,
            "p_value": None,
        }
    diffs = uma_vals - lig_vals
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return {
            "test": "wilcoxon",
            "n_paired": len(diffs),
            "n_nonzero": len(nonzero),
            "W": None,
            "p_value": None,
            "note": "insufficient nonzero pairs",
        }
    res = wilcoxon(nonzero, alternative="two-sided", method="exact")
    # Effect-size r approximation: r = Z / sqrt(N)
    # SciPy >= 1.10 returns a `zstatistic` only via method='approx'; for the
    # exact test we'll just report W and p, and let the mean-difference and
    # bootstrap CI carry the effect-size load.
    return {
        "test": "wilcoxon",
        "n_paired": len(diffs),
        "n_nonzero": len(nonzero),
        "W": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff_uma_minus_lig": float(diffs.mean()),
        "median_diff_uma_minus_lig": float(np.median(diffs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pocket_fixed_summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pocket_fixed_stats.txt",
    )
    args = parser.parse_args()

    rows = _load_summary(args.summary)
    if not rows:
        raise SystemExit("empty summary CSV")

    # Pivot: by pdb_id and method
    by_pdb: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_pdb[r["pdb_id"]][r["method"]] = r

    paired_pdbs = [pid for pid, d in by_pdb.items()
                   if "uma_v2" in d and "ligandmpnn" in d]
    logger.info("paired PDBs (both methods present): %d / %d", len(paired_pdbs), len(by_pdb))

    lines: list[str] = []
    lines.append("# Pocket-fixed redesign: UMA vs LigandMPNN")
    lines.append(f"# Paired PDBs: {len(paired_pdbs)} / {len(by_pdb)}")
    lines.append("")

    # Run each metric, overall and per-split
    for split_filter, split_name in (
        (None, "all"),
        ("metal", "metal-only"),
        ("small_molecule", "small_molecule-only"),
    ):
        if split_filter is None:
            pids = paired_pdbs
        else:
            pids = [pid for pid in paired_pdbs if by_pdb[pid]["uma_v2"]["kind"] == split_filter]
        if not pids:
            continue
        lines.append(f"## Split: {split_name}  (N = {len(pids)} PDBs)")
        lines.append("")
        for metric, metric_label in METRICS:
            uma_vals = np.array([by_pdb[pid]["uma_v2"][metric] for pid in pids])
            lig_vals = np.array([by_pdb[pid]["ligandmpnn"][metric] for pid in pids])
            valid = np.isfinite(uma_vals) & np.isfinite(lig_vals)
            uma_vals = uma_vals[valid]
            lig_vals = lig_vals[valid]
            if len(uma_vals) < 2:
                lines.append(f"### {metric_label}: insufficient valid pairs")
                continue
            stats = _wilcoxon(uma_vals, lig_vals)
            ci_lo, ci_hi = _bootstrap_ci(uma_vals - lig_vals)
            lines.append(f"### {metric_label}")
            lines.append(f"  UMA mean      = {uma_vals.mean():.4f}")
            lines.append(f"  LigandMPNN    = {lig_vals.mean():.4f}")
            lines.append(f"  diff (UMA-Lig) = {stats.get('mean_diff_uma_minus_lig', float('nan')):.4f}")
            lines.append(f"  bootstrap 95% CI on diff: [{ci_lo:+.4f}, {ci_hi:+.4f}]  (10000 resamples)")
            lines.append(f"  Wilcoxon W    = {stats.get('W')}")
            lines.append(f"  Wilcoxon p    = {stats.get('p_value')}")
            lines.append("")

    out_text = "\n".join(lines)
    args.out.write_text(out_text)
    logger.info("wrote %s", args.out)
    print(out_text)


if __name__ == "__main__":
    main()
