"""Summarise the three LigandMPNN-style interface-recovery benchmark runs.

Reads the per-PDB CSVs produced by ``scripts/benchmark_interface_recovery.py``
and prints the mean-of-per-PDB-medians for each ligand class alongside the
LigandMPNN-paper reference numbers.

Protocol (matches Dauparas et al.): for every PDB, 10 AR samples at T=0.1 →
per-sample recovery on sidechain-interface residues (≤5 Å of any nonprotein
heavy atom) → median across samples = one scalar per PDB → mean across PDBs
is the headline.

Run after the three jobs from
``scripts/SLURM/05c_benchmark_interface_recovery.sh`` finish:

    uv run python scripts/summarize_test_benchmarks.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "benchmark" / "interface_recovery"

# Classes UMA-Inverse can be fairly compared on. Nucleotide is NOT included:
# the current featurizer (src/data/pdb_parser.py) only ingests HETATM records
# as ligand context, while DNA/RNA reside in ATOM records with residue names
# DA/DC/DG/DT (DNA) or A/U/G/C (RNA). The model was therefore trained without
# any nucleotide context and cannot be compared to LigandMPNN on that split
# until the featurizer is extended and the model retrained.
COMPARED_CLASSES = ("metal", "small_molecule")
NUCLEOTIDE_CLASS = "nucleotide"
DEFAULT_RUN_PREFIX = "ep11"

# Dauparas et al. (LigandMPNN paper) — interface recovery at 5 Å sidechain
# criterion. ProteinMPNN numbers included for context.
LIGANDMPNN_REF = {
    "metal":          0.775,
    "nucleotide":     0.505,
    "small_molecule": 0.633,
}
PROTEINMPNN_REF = {
    "metal":          0.406,
    "nucleotide":     0.340,
    "small_molecule": 0.505,
}


def _load_medians(run_dir: Path) -> tuple[list[float], int, dict]:
    """Return (per-PDB median recoveries, num pdbs, summary dict)."""
    per_pdb_csv = run_dir / "per_pdb.csv"
    summary_json = run_dir / "summary.json"
    if not per_pdb_csv.exists():
        raise FileNotFoundError(f"missing per_pdb.csv: {per_pdb_csv}")

    medians: list[float] = []
    with per_pdb_csv.open() as f:
        for row in csv.DictReader(f):
            m = float(row["median_recovery"])
            if m == m:  # filter NaN
                medians.append(m)

    summary: dict = {}
    if summary_json.exists():
        summary = json.loads(summary_json.read_text())
    return medians, len(medians), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--run-prefix", type=str, default=DEFAULT_RUN_PREFIX,
        help="Prefix used by 05c_benchmark_interface_recovery.sh to name run "
             "subdirs. Default 'ep11' matches the v1 submission. Use e.g. "
             "'v2-ep19' for a v2 stage-3 epoch-19 ckpt.",
    )
    parser.add_argument(
        "--ckpt-label", type=str, default=None,
        help="Human-readable ckpt label injected into the markdown header "
             "(e.g. 'v2 stage-3 epoch 19, val_loss 1.146, val_acc 63.7%%'). "
             "Falls back to the run-prefix if omitted.",
    )
    args = parser.parse_args()

    compared_runs = {cls: f"{args.run_prefix}-test_{cls}" for cls in COMPARED_CLASSES}
    nucleotide_run = (NUCLEOTIDE_CLASS, f"{args.run_prefix}-test_{NUCLEOTIDE_CLASS}")

    rows: list[dict] = []
    missing: list[str] = []

    for cls, run_name in compared_runs.items():
        run_dir = args.out_dir / run_name
        try:
            medians, n, summary = _load_medians(run_dir)
        except FileNotFoundError as exc:
            missing.append(f"{cls}: {exc}")
            continue
        if not medians:
            missing.append(f"{cls}: no valid medians in {run_dir}")
            continue

        mean = statistics.fmean(medians)
        median = statistics.median(medians)
        stdev = statistics.pstdev(medians) if n > 1 else 0.0
        rows.append({
            "class":          cls,
            "n_pdbs":         n,
            "n_samples":      summary.get("num_samples_per_pdb", "?"),
            "temperature":    summary.get("temperature", "?"),
            "cutoff":         summary.get("cutoff_angstroms", "?"),
            "uma_mean":       mean,
            "uma_median":     median,
            "uma_stdev":      stdev,
            "lig_mpnn":       LIGANDMPNN_REF[cls],
            "p_mpnn":         PROTEINMPNN_REF[cls],
            "delta_lig_mpnn": mean - LIGANDMPNN_REF[cls],
        })

    if missing:
        print("!! missing runs:")
        for m in missing:
            print(f"   - {m}")
        if not rows:
            raise SystemExit(1)

    # Nucleotide diagnostic block — not a fair comparison, but worth surfacing
    # so the paper-facing numbers don't silently omit it.
    nucl_cls, nucl_run = nucleotide_run
    nucl_block: dict | None = None
    try:
        nucl_medians, nucl_n, nucl_summary = _load_medians(args.out_dir / nucl_run)
        if nucl_medians:
            nucl_block = {
                "n_pdbs_evaluated":    nucl_n,
                "n_pdbs_requested":    nucl_summary.get("num_pdbs_requested"),
                "n_pdbs_skipped":      nucl_summary.get("num_pdbs_skipped"),
                "mean":                statistics.fmean(nucl_medians),
                "median":              statistics.median(nucl_medians),
            }
    except FileNotFoundError:
        pass

    # Terminal table
    header = (f"{'class':16s} {'N':>5s} {'K':>3s} {'T':>5s} "
              f"{'UMA mean':>9s} {'UMA med':>9s} {'UMA σ':>7s} "
              f"{'LigMPNN':>8s} {'PMPNN':>7s} {'Δ vs Lig':>9s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['class']:16s} "
              f"{r['n_pdbs']:>5d} "
              f"{str(r['n_samples']):>3s} "
              f"{str(r['temperature']):>5s} "
              f"{r['uma_mean']:>9.3f} "
              f"{r['uma_median']:>9.3f} "
              f"{r['uma_stdev']:>7.3f} "
              f"{r['lig_mpnn']:>8.3f} "
              f"{r['p_mpnn']:>7.3f} "
              f"{r['delta_lig_mpnn']:>+9.3f}")

    # Terminal nucleotide diagnostic
    if nucl_block:
        print()
        print("nucleotide (NOT compared — featurizer limitation):")
        print(f"  evaluated {nucl_block['n_pdbs_evaluated']} / "
              f"{nucl_block['n_pdbs_requested']} PDBs "
              f"(skipped {nucl_block['n_pdbs_skipped']} — no HETATM ligand atoms)")
        print(f"  headline on evaluable subset: "
              f"mean={nucl_block['mean']:.3f}  median={nucl_block['median']:.3f}")
        print(f"  LigandMPNN ref: {LIGANDMPNN_REF[nucl_cls]:.3f}  "
              f"(not a fair comparison — see markdown report)")

    # Markdown dump for the lab notebook
    ckpt_label = args.ckpt_label or args.run_prefix
    md_lines = [
        "# UMA-Inverse vs LigandMPNN — interface sequence recovery",
        "",
        "Protocol: 10 autoregressive samples per PDB (random decoding order, "
        "T=0.1). Recovery restricted to residues whose sidechain has at least "
        "one heavy atom within 5 Å of any nonprotein heavy atom. Per-PDB "
        "statistic is the median across the 10 samples; the headline is the "
        "mean of those per-PDB medians.",
        "",
        f"UMA-Inverse checkpoint: {ckpt_label}.",
        "Reference numbers from Dauparas et al. (LigandMPNN paper).",
        "",
        "| class | N PDBs | K samples | T | **UMA mean** | UMA median | UMA σ | LigandMPNN | ProteinMPNN | Δ vs LigandMPNN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['class']} | {r['n_pdbs']} | {r['n_samples']} | {r['temperature']} "
            f"| **{r['uma_mean']:.3f}** | {r['uma_median']:.3f} | {r['uma_stdev']:.3f} "
            f"| {r['lig_mpnn']:.3f} | {r['p_mpnn']:.3f} | {r['delta_lig_mpnn']:+.3f} |"
        )
    md_lines.extend([
        "",
        "## Nucleotide split — deferred to future work",
        "",
        "The `test_nucleotide.json` split is excluded from the comparison table "
        "above because the current featurizer (`src/data/pdb_parser.py`) only "
        "treats `HETATM` records as ligand atoms. DNA and RNA are stored in "
        "PDB files as `ATOM` records with residue names `DA/DC/DG/DT` (DNA) "
        "or `A/U/G/C` (RNA), so UMA-Inverse as trained has **never seen "
        "nucleic acid as ligand context**. Any recovery we report on this "
        "split would reflect the model's performance in the absence of the "
        "relevant ligand, not its genuine nucleotide-binding design ability.",
        "",
        "Supporting nucleotide conditioning properly requires: (1) extending "
        "the featurizer to ingest nucleic acid atoms as ligand context, "
        "(2) re-preprocessing the training cache, (3) a training cycle on "
        "the updated data. Flagged as follow-up work.",
    ])
    if nucl_block:
        md_lines.extend([
            "",
            f"For reference, on the {nucl_block['n_pdbs_evaluated']} / "
            f"{nucl_block['n_pdbs_requested']} nucleotide PDBs that happened "
            f"to have HETATM ligands (e.g. metal cofactors), UMA-Inverse's "
            f"interface recovery is mean = **{nucl_block['mean']:.3f}**, "
            f"median = **{nucl_block['median']:.3f}** — but this number is "
            f"measured on non-nucleotide ligand interfaces within those PDBs "
            f"and is not directly comparable to LigandMPNN's "
            f"{LIGANDMPNN_REF[nucl_cls]:.3f} on this split.",
        ])
    md_lines.append("")
    md_text = "\n".join(md_lines)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Always write a prefix-tagged copy so historical runs aren't overwritten
    # when new ckpts are benchmarked.
    tagged_path = args.out_dir / f"test_summary_{args.run_prefix}.md"
    tagged_path.write_text(md_text)
    print(f"\nwrote {tagged_path}")
    # Also refresh the canonical 'test_summary.md' headline for the latest run.
    canonical_path = args.out_dir / "test_summary.md"
    canonical_path.write_text(md_text)
    print(f"wrote {canonical_path}")


if __name__ == "__main__":
    main()
