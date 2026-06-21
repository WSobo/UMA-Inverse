"""Inference wall-clock probe: UMA-Inverse v3 vs LigandMPNN on a fixed PDB.

Times one teacher-forced scoring forward pass per model on the same input
structure. UMA timing is in-process (model load excluded from per-call
timing); LigandMPNN timing is subprocess-based with a warmup-difference
subtraction so per-pass time is isolated from import/load overhead.

Output: ``outputs/preprint/wallclock.csv`` with columns
``model, pdb_id, num_residues, num_ligand_atoms, n_trials, median_seconds,
iqr_seconds, p05_seconds, p95_seconds``.

Usage:
    uv run python scripts/probe_inference_wallclock.py \\
        --pdb data/raw/pdb_archive/test_small_molecule/<PDB>.pdb \\
        --uma-ckpt checkpoints/uma-inverse-v3.ckpt \\
        --ligandmpnn-ckpt /path/to/ligandmpnn_v_32_010_25.pt \\
        --n-warmup 5 --n-trials 50 \\
        --out outputs/preprint/wallclock.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.decoding import score_sequence  # noqa: E402
from src.inference.session import InferenceSession  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wallclock")


def _summarize(times_s: list[float]) -> dict[str, float]:
    arr = np.asarray(times_s, dtype=float)
    return {
        "median_seconds": float(np.median(arr)),
        "iqr_seconds": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "p05_seconds": float(np.percentile(arr, 5)),
        "p95_seconds": float(np.percentile(arr, 95)),
    }


def _time_uma(
    pdb_path: Path,
    ckpt: Path,
    config: Path,
    n_warmup: int,
    n_trials: int,
) -> dict[str, float | int | str]:
    """Time UMA's autoregressive scoring per-call. Model load is excluded."""
    session = InferenceSession.from_checkpoint(config_path=config, checkpoint=ckpt)
    # max_total_nodes large enough that no cropping happens
    ctx = session.load_structure(pdb_path, max_total_nodes=5000)

    # Warmup (CUDA kernel autotune + cudnn benchmark)
    for _ in range(n_warmup):
        score_sequence(session, ctx, mode="autoregressive", num_batches=1, seed=0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        score_sequence(session, ctx, mode="autoregressive", num_batches=1, seed=i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    summary = _summarize(times)
    return {
        "model": "uma-inverse-v3",
        "pdb_id": pdb_path.stem,
        "num_residues": int(ctx.residue_count),
        "n_trials": n_trials,
        **summary,
    }


def _time_ligandmpnn(
    pdb_path: Path,
    ckpt: Path,
    ligandmpnn_dir: Path,
    micromamba_env: str,
    n_warmup: int,
    n_trials: int,
    out_dir: Path,
) -> dict[str, float | int | str]:
    """Time LigandMPNN's score.py via subprocess, isolating per-pass time.

    Strategy: run the script twice with different ``--number_of_batches``;
    the difference in wall time divided by the trial count isolates the
    per-pass cost from one-off Python/torch import + model load overhead.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir_warm = out_dir / "ligandmpnn_warm"
    score_dir_trial = out_dir / "ligandmpnn_trial"
    score_dir_warm.mkdir(parents=True, exist_ok=True)
    score_dir_trial.mkdir(parents=True, exist_ok=True)

    def _invoke(n_batches: int, score_dir: Path) -> float:
        # autoregressive_score=1 with multiple batches loops the forward
        # pass `number_of_batches` times averaging random orders, which is
        # the closest analogue to UMA's per-batch score_sequence call.
        cmd_inner = (
            f"cd '{ligandmpnn_dir}' && "
            f"python score.py "
            f"--model_type ligand_mpnn "
            f"--checkpoint_ligand_mpnn '{ckpt}' "
            f"--pdb_path '{pdb_path}' "
            f"--out_folder '{score_dir}' "
            f"--autoregressive_score 1 "
            f"--use_sequence 1 "
            f"--number_of_batches {n_batches} "
            f"--batch_size 1 "
            f"--seed 0"
        )
        cmd = [
            "bash", "-c",
            f"eval \"$(micromamba shell hook --shell bash)\" && "
            f"micromamba activate {micromamba_env} && {cmd_inner}",
        ]
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            raise RuntimeError(
                f"LigandMPNN score.py failed (n_batches={n_batches}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return elapsed

    logger.info("LigandMPNN warmup pass (n_batches=%d)...", n_warmup)
    t_warm = _invoke(n_warmup, score_dir_warm)
    logger.info("warmup wall: %.3f s", t_warm)

    logger.info("LigandMPNN trial pass (n_batches=%d)...", n_warmup + n_trials)
    t_trial = _invoke(n_warmup + n_trials, score_dir_trial)
    logger.info("trial wall: %.3f s", t_trial)

    per_pass = (t_trial - t_warm) / float(n_trials)
    # Single-point timing — express as median = mean = per_pass; IQR/percentiles
    # are unavailable from the two-invocation strategy.
    return {
        "model": "ligandmpnn",
        "pdb_id": pdb_path.stem,
        "num_residues": -1,  # unknown without parsing the .npz output
        "n_trials": n_trials,
        "median_seconds": per_pass,
        "iqr_seconds": float("nan"),
        "p05_seconds": float("nan"),
        "p95_seconds": float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True, help="PDB file to time on")
    parser.add_argument(
        "--uma-ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "uma-inverse-v3.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "old_configs" / "config_v3.yaml",
        help="Hydra config matching the ckpt's training-time flags.",
    )
    parser.add_argument(
        "--ligandmpnn-ckpt",
        type=Path,
        default=Path(
            "/private/groups/yehlab/wsobolew/01_software/LigandMPNN/"
            "model_params/ligandmpnn_v_32_010_25.pt"
        ),
    )
    parser.add_argument(
        "--ligandmpnn-dir",
        type=Path,
        default=Path("/private/groups/yehlab/wsobolew/01_software/LigandMPNN"),
    )
    parser.add_argument("--micromamba-env", type=str, default="ligandmpnn_env")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "wallclock_scratch",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "wallclock.csv",
    )
    parser.add_argument(
        "--skip-ligandmpnn",
        action="store_true",
        help="Skip LigandMPNN timing (UMA-only run)",
    )
    args = parser.parse_args()

    if not args.pdb.exists():
        raise FileNotFoundError(f"PDB not found: {args.pdb}")
    if not args.uma_ckpt.exists():
        raise FileNotFoundError(f"UMA ckpt not found: {args.uma_ckpt}")

    rows: list[dict] = []

    logger.info("timing UMA-Inverse v3 on %s ...", args.pdb)
    rows.append(
        _time_uma(
            pdb_path=args.pdb,
            ckpt=args.uma_ckpt,
            config=args.config,
            n_warmup=args.n_warmup,
            n_trials=args.n_trials,
        )
    )
    logger.info("UMA median: %.4f s", rows[-1]["median_seconds"])

    if not args.skip_ligandmpnn:
        if not args.ligandmpnn_ckpt.exists():
            logger.warning(
                "LigandMPNN ckpt not found at %s — skipping LigandMPNN timing",
                args.ligandmpnn_ckpt,
            )
        else:
            logger.info("timing LigandMPNN on %s ...", args.pdb)
            rows.append(
                _time_ligandmpnn(
                    pdb_path=args.pdb,
                    ckpt=args.ligandmpnn_ckpt,
                    ligandmpnn_dir=args.ligandmpnn_dir,
                    micromamba_env=args.micromamba_env,
                    n_warmup=args.n_warmup,
                    n_trials=args.n_trials,
                    out_dir=args.scratch_dir,
                )
            )
            logger.info("LigandMPNN per-pass: %.4f s", rows[-1]["median_seconds"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
