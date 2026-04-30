"""UMA-Inverse pocket-fixed redesign over the 20 selected PDBs.

For each PDB in outputs/preprint/pdb_selection.json:
    - Load the structure via InferenceSession (v2-ep19 ckpt by default).
    - Build DesignConstraints with fix=<pocket residue IDs>.
    - Resolve constraints; sanity-check that ResolvedConstraints.fixed_mask
      matches the JSON's pocket residues.
    - Generate K sequences autoregressively (random decoding order, T=0.1).
    - Save FASTA + per-sample distal recovery (positions in `designable_mask`,
      i.e. non-fixed) + the pairwise Hamming distance matrix at distal
      positions only (used by the diversity figure in Phase C).

Output dir layout (mirrors benchmark_interface_recovery for consistency):
    outputs/preprint/uma_pocket_fixed/{pdb_id}/
        designs.fasta                # one record per sample
        per_sample.csv               # pdb_id, sample_idx, distal_recovery, ...
        hamming_distal.npy           # KxK pairwise distances at distal positions
        meta.json                    # ckpt, settings, pocket_residue_ids, native seq
    outputs/preprint/uma_pocket_fixed/summary.csv   # one row per PDB

Run via SLURM:
    sbatch scripts/SLURM/preprint_uma_pocket_fixed.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.inference.constraints import DesignConstraints
from src.inference.decoding import autoregressive_design
from src.inference.session import InferenceSession
from src.utils.io import ID_TO_AA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("uma_pocket_fixed")


def _resolve_pdb_path(pdb_dir: Path, pdb_id: str) -> Path | None:
    pid = pdb_id.lower()
    nested = pdb_dir / pid[1:3] / f"{pid}.pdb"
    if nested.exists():
        return nested
    flat = pdb_dir / f"{pid}.pdb"
    if flat.exists():
        return flat
    return None


def _tokens_to_seq(tokens: torch.Tensor) -> str:
    return "".join(ID_TO_AA.get(int(t), "X") for t in tokens)


def _hamming_distal(samples_tokens: torch.Tensor, distal_mask: torch.Tensor) -> np.ndarray:
    """Pairwise Hamming distances at distal positions only.

    Args:
        samples_tokens: [K, L] int — K sequences over L residues.
        distal_mask: [L] bool — True where we count differences (non-fixed positions).

    Returns:
        [K, K] float — pairwise normalized Hamming distance (count differs / # distal).
    """
    K = samples_tokens.shape[0]
    n_distal = int(distal_mask.sum().item())
    if n_distal == 0:
        return np.zeros((K, K), dtype=np.float32)
    distal = samples_tokens[:, distal_mask]  # [K, n_distal]
    # Pairwise: D[i,j] = mean(distal[i] != distal[j])
    D = np.zeros((K, K), dtype=np.float32)
    distal_np = distal.cpu().numpy()
    for i in range(K):
        for j in range(i + 1, K):
            D[i, j] = (distal_np[i] != distal_np[j]).mean()
            D[j, i] = D[i, j]
    return D


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "pairmixerinv-v2-stage3-nodes384-ddp8"
        / "uma-inverse-19-1.1463.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
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
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "uma_pocket_fixed",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--decoding-order", choices=["random", "left-to-right"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of PDBs (for piloting on 1-2 first).")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(args.selection.read_text())
    entries: list[tuple[str, str, dict]] = []  # (pdb_id, kind, entry)
    for entry in selection["small_molecule"]:
        entries.append((entry["pdb_id"], "small_molecule", entry))
    for entry in selection["metal"]:
        entries.append((entry["pdb_id"], "metal", entry))
    if args.limit is not None:
        entries = entries[: args.limit]
    logger.info("loaded %d PDBs from selection", len(entries))

    logger.info("loading ckpt: %s", args.ckpt)
    session = InferenceSession.from_checkpoint(
        config_path=args.config, checkpoint=args.ckpt, device=args.device,
    )
    logger.info("device: %s", session.device)

    summary_rows: list[dict] = []
    t_start = time.time()

    for idx, (pdb_id, kind, entry) in enumerate(entries, 1):
        pdb_dir = args.smallmol_pdb_dir if kind == "small_molecule" else args.metal_pdb_dir
        pdb_path = _resolve_pdb_path(pdb_dir, pdb_id)
        if pdb_path is None:
            logger.error("PDB not found: %s in %s — skipping", pdb_id, pdb_dir)
            continue

        run_out_dir = args.out_dir / pdb_id
        run_out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            ctx = session.load_structure(pdb_path, max_total_nodes=2000)
        except Exception as exc:
            logger.error("load_structure failed for %s: %s — skipping", pdb_id, exc)
            continue

        # Build constraints. The pocket_residues from the selection JSON were
        # extracted via BioPython; the rid format ("A23") matches what
        # InferenceSession.load_structure produces.
        pocket_residues = set(entry["pocket_residues"])
        # Filter to only residues actually present in the ctx — some pocket
        # residues from the BioPython parse may be on chains UMA's parser
        # didn't ingest (e.g. nucleotide chains in the metal split).
        present = pocket_residues & set(ctx.residue_ids)
        missing = pocket_residues - present
        if missing:
            logger.info(
                "%s: %d/%d pocket residues not in UMA context (likely non-protein chains): %s",
                pdb_id, len(missing), len(pocket_residues), sorted(missing),
            )

        constraints = DesignConstraints(fix=present)
        try:
            resolved = constraints.resolve(ctx)
        except Exception as exc:
            logger.error("constraint resolve failed for %s: %s — skipping", pdb_id, exc)
            continue

        n_fixed = int(resolved.fixed_mask.sum().item())
        n_designable = int(resolved.designable_mask.sum().item())
        logger.info(
            "%s [%s]: L=%d, fixed=%d, designable=%d  (kind=%s, ligand=%s)",
            pdb_id, kind, ctx.residue_count, n_fixed, n_designable, kind,
            entry.get("ccd_code") or entry.get("ion"),
        )
        if n_designable == 0:
            logger.error("%s: all positions fixed — nothing to redesign, skipping", pdb_id)
            continue

        # Sample K sequences.
        per_pdb_seed = args.seed + 100 * idx  # offset so each PDB has independent seeds
        samples = autoregressive_design(
            session=session,
            ctx=ctx,
            constraints=resolved,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            temperature=args.temperature,
            seed=per_pdb_seed,
            decoding_order=args.decoding_order,
        )

        native = ctx.native_sequence.cpu()
        designable_mask_cpu = resolved.designable_mask.cpu()
        fixed_mask_cpu = resolved.fixed_mask.cpu()

        # Sanity check: sampled tokens at fixed positions must equal native.
        for s_idx, s in enumerate(samples):
            fixed_match = ((s.token_ids[fixed_mask_cpu] == native[fixed_mask_cpu])
                           | ~fixed_mask_cpu[fixed_mask_cpu]).all()
            if not fixed_match:
                # This shouldn't happen given how decoding handles fixed positions
                # but we surface it for debugging.
                logger.warning(
                    "%s sample %d: fixed-mask sanity check failed (decoder produced "
                    "non-native at a fixed position!)",
                    pdb_id, s_idx,
                )

        # Per-sample metrics
        per_sample_rows = []
        all_tokens = torch.stack([s.token_ids for s in samples])  # [K, L]
        for s_idx, s in enumerate(samples):
            pred = s.token_ids
            # Distal recovery = correct fraction over designable positions
            valid = designable_mask_cpu & (native != 20)
            if valid.any():
                correct = ((pred == native) & valid).sum().item() / valid.sum().item()
            else:
                correct = float("nan")
            # Pocket recovery (sanity: should be 1.0 since fixed)
            pocket_valid = fixed_mask_cpu & (native != 20)
            if pocket_valid.any():
                pocket_correct = ((pred == native) & pocket_valid).sum().item() / pocket_valid.sum().item()
            else:
                pocket_correct = float("nan")
            per_sample_rows.append({
                "pdb_id": pdb_id,
                "kind": kind,
                "sample_idx": s_idx,
                "seed": s.seed,
                "distal_recovery": correct,
                "pocket_recovery": pocket_correct,
                "overall_confidence": s.overall_confidence(designable_mask_cpu),
                "n_designable": n_designable,
                "n_fixed": n_fixed,
            })

        # Pairwise Hamming distance at distal positions
        H = _hamming_distal(all_tokens, designable_mask_cpu)
        np.save(run_out_dir / "hamming_distal.npy", H)

        # Per-sample CSV
        with (run_out_dir / "per_sample.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            for r in per_sample_rows:
                writer.writerow(r)

        # FASTA: one record per sample
        with (run_out_dir / "designs.fasta").open("w") as f:
            for s_idx, s in enumerate(samples):
                seq_str = _tokens_to_seq(s.token_ids)
                f.write(f">{pdb_id}_sample{s_idx:02d}_seed{s.seed} distal_recovery={per_sample_rows[s_idx]['distal_recovery']:.4f}\n")
                f.write(seq_str + "\n")

        # Meta JSON
        meta = {
            "pdb_id": pdb_id,
            "kind": kind,
            "ligand_ccd_or_ion": entry.get("ccd_code") or entry.get("ion"),
            "n_residues": ctx.residue_count,
            "n_fixed": n_fixed,
            "n_designable": n_designable,
            "missing_pocket_residues": sorted(missing),
            "native_sequence": _tokens_to_seq(native),
            "ckpt": str(args.ckpt),
            "config": str(args.config),
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "decoding_order": args.decoding_order,
            "seed_base": per_pdb_seed,
            "git_hash": _git_hash(),
        }
        (run_out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Summary aggregation
        distal_recoveries = [r["distal_recovery"] for r in per_sample_rows]
        upper = np.triu_indices(args.num_samples, k=1)
        mean_pairwise_hamming = float(H[upper].mean()) if H.size else float("nan")
        summary_rows.append({
            "pdb_id": pdb_id,
            "kind": kind,
            "method": "uma_v2",
            "n_residues": ctx.residue_count,
            "n_fixed": n_fixed,
            "n_designable": n_designable,
            "mean_distal_recovery": float(np.mean(distal_recoveries)),
            "median_distal_recovery": float(np.median(distal_recoveries)),
            "stdev_distal_recovery": float(np.std(distal_recoveries, ddof=0)),
            "mean_pairwise_hamming_distal": mean_pairwise_hamming,
            "wall_seconds": time.time() - t0,
        })

        wall = time.time() - t0
        logger.info(
            "%s done in %.1fs   distal_recovery mean=%.3f   hamming_diversity=%.3f",
            pdb_id, wall, summary_rows[-1]["mean_distal_recovery"], mean_pairwise_hamming,
        )

    # Write summary CSV
    if summary_rows:
        summary_csv = args.out_dir / "summary.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        logger.info("wrote summary: %s  (%d rows)", summary_csv, len(summary_rows))

    # Top-level meta
    args.out_dir.joinpath("run_meta.json").write_text(json.dumps({
        "method": "uma_v2",
        "ckpt": str(args.ckpt),
        "ckpt_stem": args.ckpt.stem,
        "num_samples_per_pdb": args.num_samples,
        "temperature": args.temperature,
        "decoding_order": args.decoding_order,
        "git_hash": _git_hash(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "n_pdbs": len(summary_rows),
        "total_wall_seconds": time.time() - t_start,
    }, indent=2))

    logger.info("done in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
