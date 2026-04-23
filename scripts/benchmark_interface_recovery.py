"""LigandMPNN-style interface sequence-recovery benchmark.

Reproduces the Dauparas et al. protocol:

    for each PDB in --val-json:
        encode structure once
        generate --num-samples sequences autoregressively
            (random decoding order, temperature=--temperature)
        compute per-sample recovery restricted to sidechain-interface residues
            (sidechain heavy atom within --cutoff Å of any nonprotein heavy atom)
        record the median recovery across samples → one scalar per PDB
    aggregate per-PDB medians (mean for the headline, distribution for violins)

Output per run:
    outputs/benchmark/interface_recovery/<run_name>/per_pdb.csv       (one row per PDB)
    outputs/benchmark/interface_recovery/<run_name>/per_sample.csv    (per-PDB × per-sample)
    outputs/benchmark/interface_recovery/<run_name>/summary.json      (headline stats + config)

Run via SLURM wrapper:

    bash scripts/SLURM/05c_benchmark_interface_recovery.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.benchmarks.interface_mask import compute_sidechain_interface_mask
from src.data.ligandmpnn_bridge import load_json_ids, resolve_pdb_path
from src.inference.constraints import DesignConstraints
from src.inference.decoding import autoregressive_design
from src.inference.session import InferenceSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("interface_recovery")


@dataclass
class PerPdbRow:
    pdb_id: str
    num_residues: int
    num_interface: int
    median_recovery: float          # median across the N samples
    mean_recovery: float            # mean across samples (for variance sanity)
    std_recovery: float
    min_recovery: float
    max_recovery: float
    median_overall_recovery: float  # same stats on whole-protein recovery
    wall_seconds: float


def _per_sample_interface_recovery(
    pred_tokens: torch.Tensor,
    native_tokens: torch.Tensor,
    interface_mask: torch.Tensor,
) -> float:
    """Fraction of interface positions where pred == native (excluding X)."""
    valid = (native_tokens != 20) & interface_mask
    if not valid.any():
        return float("nan")
    correct = (pred_tokens == native_tokens) & valid
    return correct.sum().item() / valid.sum().item()


def _per_sample_overall_recovery(
    pred_tokens: torch.Tensor,
    native_tokens: torch.Tensor,
) -> float:
    valid = native_tokens != 20
    if not valid.any():
        return float("nan")
    correct = (pred_tokens == native_tokens) & valid
    return correct.sum().item() / valid.sum().item()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument("--val-json", required=True, type=Path,
                        help="LigandMPNN-style JSON list of PDB IDs (e.g. test_metal.json).")
    parser.add_argument("--pdb-dir", required=True, type=Path,
                        help="Directory with RCSB-style <xx>/<id>.pdb layout or flat <id>.pdb.")
    parser.add_argument("--run-name", required=True, type=str,
                        help="Subdirectory name under --out-dir.")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "benchmark" / "interface_recovery")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Sequences generated per PDB (default: 10).")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Samples decoded in parallel per forward pass. "
                             "Pair-tensor memory scales with batch_size × L². "
                             "Lower this for large structures; raise for speed "
                             "on small ones (default: 2, safe up to L~2000 on 24 GB).")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="Sidechain-to-ligand distance threshold in Å.")
    parser.add_argument("--decoding-order", choices=["random", "left-to-right"],
                        default="random")
    parser.add_argument("--max-total-nodes", type=int, default=5000,
                        help="Residue-crop budget (large → no cropping for most PDBs).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of PDBs (debugging).")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    args = parser.parse_args()

    out_dir = args.out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pdb_csv = out_dir / "per_pdb.csv"
    per_sample_csv = out_dir / "per_sample.csv"
    summary_json = out_dir / "summary.json"

    logger.info("run dir      : %s", out_dir)
    logger.info("val json     : %s", args.val_json)
    logger.info("pdb dir      : %s", args.pdb_dir)
    logger.info("ckpt         : %s", args.ckpt)
    logger.info("samples/pdb  : %d   T=%.2f   cutoff=%.1fÅ", args.num_samples,
                args.temperature, args.cutoff)

    session = InferenceSession.from_checkpoint(
        config_path=args.config, checkpoint=args.ckpt, device=args.device,
    )
    logger.info("device       : %s", session.device)

    pdb_ids = load_json_ids(str(args.val_json))
    if args.limit is not None:
        pdb_ids = pdb_ids[: args.limit]
    logger.info("pdb count    : %d", len(pdb_ids))

    # Resolve paths up front so we can skip missing ones cleanly.
    resolved: list[tuple[str, Path]] = []
    for pid in pdb_ids:
        p = resolve_pdb_path(str(args.pdb_dir), pid)
        if p is None:
            logger.warning("pdb not found in archive: %s", pid)
            continue
        resolved.append((pid, Path(p)))
    logger.info("resolved     : %d / %d PDBs", len(resolved), len(pdb_ids))

    unconstrained = DesignConstraints.from_cli()

    per_pdb_rows: list[PerPdbRow] = []
    per_sample_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    t_start = time.time()
    for idx, (pid, path) in enumerate(resolved, 1):
        t0 = time.time()
        try:
            ctx = session.load_structure(path, max_total_nodes=args.max_total_nodes)
        except Exception as exc:
            logger.warning("skip %s: load_structure failed (%s)", pid, exc)
            skipped.append((pid, f"load: {exc!r}"))
            continue

        try:
            interface = compute_sidechain_interface_mask(
                path, ctx.residue_ids, cutoff=args.cutoff,
            )
        except Exception as exc:
            logger.warning("skip %s: interface mask failed (%s)", pid, exc)
            skipped.append((pid, f"mask: {exc!r}"))
            continue

        n_interface = int(interface.sum().item())
        if n_interface == 0:
            logger.debug("%s: zero interface residues — skipping", pid)
            skipped.append((pid, "no interface residues"))
            continue

        resolved_constraints = unconstrained.resolve(ctx)
        samples = autoregressive_design(
            session=session,
            ctx=ctx,
            constraints=resolved_constraints,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            temperature=args.temperature,
            seed=args.seed + idx,
            decoding_order=args.decoding_order,
        )

        native = ctx.native_sequence.cpu()
        sample_recoveries: list[float] = []
        sample_overall: list[float] = []
        for j, s in enumerate(samples):
            pred = s.token_ids  # already on CPU from autoregressive_design
            iface_rec = _per_sample_interface_recovery(pred, native, interface)
            over_rec = _per_sample_overall_recovery(pred, native)
            sample_recoveries.append(iface_rec)
            sample_overall.append(over_rec)
            per_sample_rows.append({
                "pdb_id": pid,
                "sample_idx": j,
                "seed": s.seed,
                "interface_recovery": iface_rec,
                "overall_recovery": over_rec,
            })

        wall = time.time() - t0
        row = PerPdbRow(
            pdb_id=pid,
            num_residues=ctx.residue_count,
            num_interface=n_interface,
            median_recovery=statistics.median(sample_recoveries),
            mean_recovery=statistics.fmean(sample_recoveries),
            std_recovery=statistics.pstdev(sample_recoveries) if len(sample_recoveries) > 1 else 0.0,
            min_recovery=min(sample_recoveries),
            max_recovery=max(sample_recoveries),
            median_overall_recovery=statistics.median(sample_overall),
            wall_seconds=wall,
        )
        per_pdb_rows.append(row)

        if idx % 20 == 0 or idx == len(resolved):
            elapsed = time.time() - t_start
            logger.info("progress %d/%d  median_iface=%.3f  (%.1fs/pdb)",
                        idx, len(resolved), row.median_recovery, wall)

    # ── Write outputs ────────────────────────────────────────────────────────
    if per_pdb_rows:
        with per_pdb_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(per_pdb_rows[0]).keys()))
            writer.writeheader()
            for r in per_pdb_rows:
                writer.writerow(asdict(r))
        logger.info("wrote %s  (%d rows)", per_pdb_csv, len(per_pdb_rows))

    if per_sample_rows:
        with per_sample_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_rows)
        logger.info("wrote %s  (%d rows)", per_sample_csv, len(per_sample_rows))

    # Headline summary
    medians = [r.median_recovery for r in per_pdb_rows if r.median_recovery == r.median_recovery]
    if medians:
        summary = {
            "run_name": args.run_name,
            "val_json": str(args.val_json),
            "pdb_dir": str(args.pdb_dir),
            "ckpt": str(args.ckpt),
            "ckpt_stem": args.ckpt.stem,
            "config": str(args.config),
            "num_samples_per_pdb": args.num_samples,
            "temperature": args.temperature,
            "cutoff_angstroms": args.cutoff,
            "decoding_order": args.decoding_order,
            "seed": args.seed,
            "num_pdbs_requested": len(pdb_ids),
            "num_pdbs_resolved": len(resolved),
            "num_pdbs_evaluated": len(per_pdb_rows),
            "num_pdbs_skipped": len(skipped),
            "skipped": skipped,
            # Headline stats on the per-PDB medians — this is the direct
            # analog of LigandMPNN's paper numbers.
            "mean_of_per_pdb_medians": statistics.fmean(medians),
            "median_of_per_pdb_medians": statistics.median(medians),
            "stdev_of_per_pdb_medians": statistics.pstdev(medians) if len(medians) > 1 else 0.0,
            "total_wall_seconds": time.time() - t_start,
            "git_hash": _git_hash(),
            "hostname": socket.gethostname(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        }
        summary_json.write_text(json.dumps(summary, indent=2))
        logger.info("headline: mean_of_per_pdb_medians = %.3f  (N=%d PDBs, %d samples each)",
                    summary["mean_of_per_pdb_medians"], len(per_pdb_rows), args.num_samples)
    else:
        logger.error("no per-PDB rows written; check skipped list")
        sys.exit(1)

    logger.info("done in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
