"""Run after the extended cofold finishes: re-run all pocket-fixed metric and
figure scripts against the COMBINED N=25 small_mol + 10 metal selection.

Steps:
  1. Pocket-fixed metrics on combined selection
        -> outputs/preprint/pocket_fixed_metrics.csv      (overwritten)
        -> outputs/preprint/pocket_fixed_summary.csv      (overwritten)
        -> outputs/preprint/pocket_fixed_aa_freq.csv      (overwritten)
  2. Pocket-fixed Wilcoxon stats on combined
        -> outputs/preprint/pocket_fixed_stats.txt        (overwritten)
  3. Cofold metrics for BOTH original (cofold/) and extended (cofold_extended/)
     trees, then concatenate into a single CSV.
        -> outputs/preprint/cofold_metrics.csv            (combined)
  4. Regenerate fig 5 (pocket-fixed) and fig 6 (cofold) from combined CSVs.

This script is designed to be safely re-runnable: each step takes a few
seconds and writes deterministic outputs. Designed for SLURM wrapper
scripts/SLURM/preprint_finalize_combined.sh which gates on the original and
extended cofold jobs both finishing successfully.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("finalize_combined")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=True)


def _concat_csvs(inputs: list[Path], output: Path) -> int:
    """Concatenate CSVs, preserving the first header. Returns total row count."""
    if not inputs:
        raise FileNotFoundError("no inputs to concatenate")
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", newline="") as out_fh:
        writer = None
        for src in inputs:
            if not src.exists():
                logger.warning("missing input %s -- skipping", src)
                continue
            with src.open() as in_fh:
                reader = csv.DictReader(in_fh)
                if writer is None:
                    writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-selection",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pdb_selection_combined.json",
    )
    parser.add_argument(
        "--extended-selection",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "pdb_selection_extended.json",
    )
    args = parser.parse_args()

    if not args.combined_selection.exists():
        raise SystemExit(
            f"missing {args.combined_selection}. "
            "Run scripts/preprint/select_extended_smallmol_pdbs.py first."
        )

    # ── 1. Pocket-fixed metrics on combined ────────────────────────────────
    logger.info("[1/4] pocket-fixed metrics (combined)")
    _run(["uv", "run", "python", "scripts/preprint/compute_pocket_fixed_metrics.py",
          "--selection", str(args.combined_selection)])

    # ── 2. Wilcoxon stats on combined ──────────────────────────────────────
    logger.info("[2/4] pocket-fixed stats (combined)")
    _run(["uv", "run", "python", "scripts/preprint/pocket_fixed_stats.py"])

    # ── 3. Cofold metrics for both trees, then concat ──────────────────────
    logger.info("[3/4] cofold metrics (original + extended)")
    out_orig = PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics_original.csv"
    out_ext = PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics_extended.csv"

    _run([
        "uv", "run", "python", "scripts/preprint/cofold_metrics.py",
        "--sampling-record", str(PROJECT_ROOT / "outputs" / "preprint"
                                  / "boltz_inputs" / "cofold" / "sampling_record.json"),
        "--cofold-base", str(PROJECT_ROOT / "outputs" / "preprint" / "cofold"),
        "--selection", str(args.combined_selection),
        "--out", str(out_orig),
    ])

    sampling_ext = (PROJECT_ROOT / "outputs" / "preprint"
                    / "boltz_inputs" / "cofold_extended" / "sampling_record.json")
    if sampling_ext.exists():
        _run([
            "uv", "run", "python", "scripts/preprint/cofold_metrics.py",
            "--sampling-record", str(sampling_ext),
            "--cofold-base", str(PROJECT_ROOT / "outputs" / "preprint" / "cofold_extended"),
            "--selection", str(args.combined_selection),
            "--out", str(out_ext),
        ])
    else:
        logger.warning("extended sampling record missing; skipping extended cofold metrics")

    # Cofold rerun (the 4 PDBs whose initial extended cofolds failed because
    # numeric CCD codes were parsed as ints). If present, results overlap
    # with extended/ for those PDBs (which had no successful outputs there),
    # so the union is non-redundant.
    out_rerun = PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics_rerun.csv"
    sampling_rerun = (PROJECT_ROOT / "outputs" / "preprint"
                      / "boltz_inputs" / "cofold_rerun" / "sampling_record.json")
    if sampling_rerun.exists():
        _run([
            "uv", "run", "python", "scripts/preprint/cofold_metrics.py",
            "--sampling-record", str(sampling_rerun),
            "--cofold-base", str(PROJECT_ROOT / "outputs" / "preprint" / "cofold_rerun"),
            "--selection", str(args.combined_selection),
            "--out", str(out_rerun),
        ])

    combined = PROJECT_ROOT / "outputs" / "preprint" / "cofold_metrics.csv"
    inputs = [p for p in (out_orig, out_ext, out_rerun) if p.exists()]
    total = _concat_csvs(inputs, combined)
    logger.info("wrote combined cofold metrics: %s  (%d rows)", combined, total)

    # ── 4. Regenerate fig 5 + fig 6 ────────────────────────────────────────
    logger.info("[4/4] regenerate fig 5 + fig 6")
    _run(["uv", "run", "python", "scripts/preprint/figures/fig5_pocket_distal.py"])
    _run(["uv", "run", "python", "scripts/preprint/figures/fig6_cofold.py"])

    print("\n=== Final summary ===")
    summary_csv = PROJECT_ROOT / "outputs" / "preprint" / "pocket_fixed_summary.csv"
    if summary_csv.exists():
        from collections import defaultdict
        rows = list(csv.DictReader(summary_csv.open()))
        by_split_method: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            by_split_method[(r["kind"], r["method"])].append(r)
        for (kind, method), rs in sorted(by_split_method.items()):
            distals = [float(r["mean_distal_recovery"]) for r in rs]
            hammings = [float(r["mean_pairwise_hamming_distal"]) for r in rs]
            print(f"  {kind:16s} {method:12s} N={len(rs):>2d}  "
                  f"distal_recovery={sum(distals)/len(distals):.3f}  "
                  f"hamming={sum(hammings)/len(hammings):.3f}")


if __name__ == "__main__":
    main()
