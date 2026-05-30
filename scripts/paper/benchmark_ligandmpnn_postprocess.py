"""Post-process LigandMPNN --save_stats .pt files into a benchmark summary.

Reads one stats .pt file per PDB produced by LigandMPNN run.py with
--save_stats 1.  For each PDB:
  - Resolves the original PDB file and re-parses it to get Cα coordinates
    and ligand atom coordinates.
  - Computes per-residue distance to the nearest ligand heavy atom (Cα to
    ligand, same metric as the UMA-Inverse benchmark).
  - Records per-position predictions so the shell-recovery comparison
    script can compare UMA vs LigandMPNN at each distance shell.

Outputs (in --out-dir):
  per_pdb.csv         one row per PDB (pdb_id, recovery, n_residues)
  per_position.parquet  per-residue table matching UMA benchmark schema
                        columns: pdb_id, position, native_token, pred_token,
                                 native_aa, pred_aa, correct,
                                 distance_to_ligand
  shell_recovery.csv  recovery binned by distance-to-ligand shell
  summary.json / summary.md  headline numbers

NOTE on evaluation mode: LigandMPNN decodes autoregressively at T=0.1.
UMA-Inverse's 0.695 is teacher-forced (argmax given full native context).
These differ systematically — teacher-forced is higher.  Use the AR UMA
benchmark (v3-ar-T0.1) for the apples-to-apples recovery comparison;
use the per_position.parquet for the per-shell comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.metrics import residue_ligand_distances
from src.data.pdb_parser import parse_pdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lmpnn_postprocess")

_INT_TO_AA = list("ACDEFGHIKLMNPQRSTVWYX")
_X_TOKEN = 20

SHELLS = [
    (0.0, 5.0,  "0-5A"),
    (5.0, 10.0, "5-10A"),
    (10.0, 15.0, "10-15A"),
    (15.0, 20.0, "15-20A"),
    (20.0, float("inf"), ">20A"),
]


def _resolve_pdb(pdb_dir: Path, pdb_id: str) -> Path | None:
    pid = pdb_id.lower()
    for p in [pdb_dir / pid[1:3] / f"{pid}.pdb", pdb_dir / f"{pid}.pdb"]:
        if p.exists():
            return p
    return None


def _get_coords(pdb_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ca_coords [L, 3], ligand_coords [M, 3]) from parse_pdb."""
    parsed = parse_pdb(str(pdb_path))
    x = torch.as_tensor(parsed["X"], dtype=torch.float32)
    mask = torch.as_tensor(parsed["mask"], dtype=torch.bool)
    ca = x[:, 1, :][mask]                       # Cα only, valid residues

    y = torch.as_tensor(parsed["Y"], dtype=torch.float32)
    y_m = torch.as_tensor(parsed["Y_m"], dtype=torch.bool)
    lig = y[y_m]

    return ca, lig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, required=True,
                        help="Directory containing LigandMPNN stats/*.pt files.")
    parser.add_argument("--pdb-dir", type=Path, required=True,
                        help="PDB archive root (same as --pdb-dir in the benchmark).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory (will be created).")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Which generated sample to evaluate (0 = first).")
    args = parser.parse_args()

    stats_files = sorted(args.stats_dir.glob("*.pt"))
    if not stats_files:
        raise SystemExit(f"No .pt files found in {args.stats_dir}")
    logger.info("found %d stats files", len(stats_files))

    per_pdb_rows: list[dict] = []
    per_pos_rows: list[dict] = []

    for pt_path in stats_files:
        pdb_id = pt_path.stem.split(".")[0]

        try:
            d = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning("%s: failed to load stats -- %s", pdb_id, e)
            continue

        native: torch.Tensor = d["native_sequence"]       # [L_total]
        generated: torch.Tensor = d["generated_sequences"] # [n_samples, L_total]
        mask: torch.Tensor = (d["mask"] * d["chain_mask"]).bool()  # [L_total]

        s_idx = min(args.sample_idx, generated.shape[0] - 1)
        gen = generated[s_idx]

        # ── Distances ──────────────────────────────────────────────────────────
        pdb_path = _resolve_pdb(args.pdb_dir, pdb_id)
        distances: torch.Tensor | None = None
        if pdb_path is not None:
            try:
                ca_coords, lig_coords = _get_coords(pdb_path)
                n_valid_pdb = ca_coords.shape[0]
                n_valid_stats = int(mask.sum().item())
                if n_valid_pdb == n_valid_stats:
                    distances = residue_ligand_distances(ca_coords, lig_coords)
                else:
                    logger.warning(
                        "%s: residue count mismatch (parse_pdb=%d, stats=%d) "
                        "-- distances will be inf",
                        pdb_id, n_valid_pdb, n_valid_stats,
                    )
            except Exception as e:
                logger.warning("%s: PDB parse failed -- %s", pdb_id, e)
        else:
            logger.warning("%s: PDB not found in %s -- distances will be inf",
                           pdb_id, args.pdb_dir)

        # ── Per-position records ───────────────────────────────────────────────
        n_correct = 0
        n_valid = 0
        dist_idx = 0  # index into distances (valid positions only)
        for pos in range(len(native)):
            if not mask[pos]:
                continue
            nat = int(native[pos].item())
            if nat == _X_TOKEN:
                dist_idx += 1
                continue
            pred = int(gen[pos].item())
            correct = int(pred == nat)
            dist = float(distances[dist_idx].item()) if distances is not None else float("inf")
            per_pos_rows.append({
                "pdb_id": pdb_id,
                "position": pos,
                "native_token": nat,
                "pred_token": pred,
                "native_aa": _INT_TO_AA[nat],
                "pred_aa": _INT_TO_AA[pred],
                "correct": correct,
                "distance_to_ligand": dist,
            })
            n_correct += correct
            n_valid += 1
            dist_idx += 1

        recovery = n_correct / n_valid if n_valid > 0 else float("nan")
        per_pdb_rows.append({"pdb_id": pdb_id, "recovery": recovery, "n_residues": n_valid})

    if not per_pdb_rows:
        raise SystemExit("no PDBs processed successfully")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── per_pdb.csv ───────────────────────────────────────────────────────────
    with (args.out_dir / "per_pdb.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "recovery", "n_residues"])
        w.writeheader()
        w.writerows(per_pdb_rows)

    # ── per_position.parquet ──────────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.DataFrame(per_pos_rows)
        df.to_parquet(args.out_dir / "per_position.parquet", index=False)
        logger.info("wrote per_position.parquet (%d rows)", len(df))
    except ImportError:
        logger.warning("pandas not available -- skipping per_position.parquet")
        df = None

    # ── shell_recovery.csv ────────────────────────────────────────────────────
    shell_rows: list[dict] = []
    for lo, hi, label in SHELLS:
        subset = [r for r in per_pos_rows
                  if lo <= r["distance_to_ligand"] < hi]
        if not subset:
            continue
        rec = sum(r["correct"] for r in subset) / len(subset)
        shell_rows.append({
            "shell": label, "lo_A": lo, "hi_A": hi,
            "n_residues": len(subset),
            "recovery": rec,
        })
    with (args.out_dir / "shell_recovery.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shell", "lo_A", "hi_A", "n_residues", "recovery"])
        w.writeheader()
        w.writerows(shell_rows)

    # ── summary ───────────────────────────────────────────────────────────────
    recs = [r["recovery"] for r in per_pdb_rows if r["recovery"] == r["recovery"]]
    pooled = sum(r["correct"] for r in per_pos_rows) / max(len(per_pos_rows), 1)
    summary = {
        "model": "LigandMPNN ligandmpnn_v_32_010_25",
        "eval_mode": "autoregressive T=0.1 (NOT teacher-forced)",
        "n_pdbs": len(per_pdb_rows),
        "n_residues": len(per_pos_rows),
        "pooled_recovery": pooled,
        "per_pdb_recovery_mean": statistics.mean(recs),
        "per_pdb_recovery_std": statistics.stdev(recs) if len(recs) > 1 else 0.0,
        "per_pdb_recovery_median": statistics.median(recs),
        "shell_recovery": {r["shell"]: r["recovery"] for r in shell_rows},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    shell_table = "\n".join(
        f"| {r['shell']:8s} | {r['recovery']:.3f} | {r['n_residues']:>7d} |"
        for r in shell_rows
    )
    md = f"""# LigandMPNN benchmark summary

> **Eval mode**: autoregressive T=0.1 (NOT teacher-forced).
> Compare AR recovery to UMA v3 `outputs/benchmark/v3-ar-T0.1/`.
> Compare per-shell recovery to UMA v3 `outputs/benchmark/v3-final/per_position.parquet`.

## Headline

- **PDBs evaluated**: {len(per_pdb_rows)}
- **Residues evaluated**: {len(per_pos_rows)}
- **Pooled sequence recovery**: {pooled:.4f}
- **Per-PDB recovery**: {statistics.mean(recs):.4f} ± {statistics.stdev(recs) if len(recs)>1 else 0:.4f} (median {statistics.median(recs):.4f})

## Recovery by distance-to-ligand shell

| Shell    | Recovery | N residues |
|----------|----------|------------|
{shell_table}
"""
    (args.out_dir / "summary.md").write_text(md)

    print(f"\nLigandMPNN benchmark ({len(per_pdb_rows)} PDBs, {len(per_pos_rows)} residues)")
    print(f"  Pooled recovery:    {pooled:.4f}")
    print(f"  Per-PDB median:     {statistics.median(recs):.4f}")
    print("\nShell recovery:")
    for r in shell_rows:
        print(f"  {r['shell']:8s}: {r['recovery']:.4f}  (n={r['n_residues']})")
    print("\n[NOTE] Autoregressive T=0.1 — use v3-ar-T0.1 for AR UMA comparison.")


if __name__ == "__main__":
    main()
