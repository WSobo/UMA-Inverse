"""Layer 3: distal-residue ligand-conditioning shift, UMA-v3 vs LigandMPNN.

The hypothesis behind the paper's "intelligent distal sequence design" angle
is that UMA's dense pair tensor propagates ligand context to distal residues
more directly than LigandMPNN's K=48 KNN graph, where the ligand signal must
percolate through 2-3 message-passing hops to reach far-from-pocket positions.

This script makes that mechanism measurable. For each PDB and each model, we
run teacher-forced AR scoring twice — once with the ligand visible, once with
ligand atoms zeroed — and capture per-position 20-AA distributions for both
conditions. Per-position KL(P_lig || P_nolig) tells us how much the model's
prediction moves when ligand context is removed; binning by distance to the
nearest ligand atom shows whether that movement is concentrated at the pocket
(LigandMPNN-style) or extends further out (the v3 prediction).

Output: one parquet row per (model, pdb, residue), with kl_shift and
distance_to_ligand. A summary CSV gives per-distance-bin mean KL.

Usage:
    uv run python scripts/paper/distal_kl_shift.py \\
        --uma-ckpt checkpoints/uma-inverse-v3.ckpt \\
        --pdb-list outputs/preprint/distal_kl/pdb_list.txt \\
        --pdb-dir data/raw/pdb_archive \\
        --out-dir outputs/preprint/distal_kl/<run_name>
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.metrics import residue_ligand_distances  # noqa: E402
from src.data.ligandmpnn_bridge import resolve_pdb_path  # noqa: E402
from src.data.pdb_parser import parse_pdb  # noqa: E402
from src.inference.session import InferenceSession, StructureContext  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("distal_kl")

DISTANCE_BINS: list[tuple[str, float, float]] = [
    ("0-5", 0.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-15", 10.0, 15.0),
    ("15-25", 15.0, 25.0),
    (">25", 25.0, float("inf")),
]


# ─── KL helper ────────────────────────────────────────────────────────────────


def _kl_per_position(
    p_lig: torch.Tensor, p_nolig: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Per-position KL(p_lig || p_nolig) over the 20-AA distribution.

    Token 20 (X) is excluded from both sides — UMA always masks it to -inf
    (so its prob is 0 by construction); LigandMPNN's softmax includes all 21
    tokens but X carries vanishing mass for any structured input. Excluding
    it explicitly avoids 0*log(0/0) numeric noise.
    """
    p = p_lig[..., :20].clamp_min(eps)
    q = p_nolig[..., :20].clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (p * (p.log() - q.log())).sum(dim=-1)  # [L]


# ─── UMA full-distribution scorer ─────────────────────────────────────────────


@torch.no_grad()
def _uma_full_distribution(
    session: InferenceSession,
    ctx: StructureContext,
    *,
    num_batches: int = 10,
    seed: int = 0,
) -> torch.Tensor:
    """Per-position 20-AA *probability* distribution averaged over decoding orders.

    Mirrors :func:`src.inference.decoding._score_autoregressive` but keeps the
    full softmax distribution at every position instead of gathering at the
    native sequence. Same numeric setup (token 20 masked to -inf before
    softmax) so probabilities sum to 1 over 20 valid tokens.

    Returns ``[L, 21]`` tensor of probabilities (token 20 column is 0).
    """
    device = ctx.device
    L = ctx.residue_count
    model = session.model

    acc_probs = torch.zeros(L, 21, device=device)
    for b in range(num_batches):
        g = torch.Generator(device="cpu").manual_seed(int(seed + b))
        perm = torch.randperm(L, generator=g).to(device)
        ranks = torch.empty(L, dtype=torch.long, device=device)
        ranks[perm] = torch.arange(L, device=device)

        seq_batched = ctx.native_sequence.unsqueeze(0)
        ar_ctx = model._autoregressive_context(
            z=ctx.z,
            sequence=seq_batched,
            residue_mask=ctx.residue_mask,
            decoding_order=ranks.unsqueeze(0),
        )
        decoder_input = torch.cat([ctx.node_repr_res, ar_ctx, ctx.lig_ctx], dim=-1)
        logits = model.decoder(decoder_input)[0]
        logits = logits.clone()
        logits[:, 20] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        acc_probs = acc_probs + probs

    return acc_probs / float(num_batches)


# ─── Per-PDB UMA evaluation ───────────────────────────────────────────────────


def _eval_uma_one_pdb(
    session: InferenceSession,
    pdb_path: Path,
    *,
    num_batches: int,
    seed: int,
) -> list[dict] | None:
    """Run UMA twice on the same PDB (ligand-on / ligand-off) and emit per-pos rows.

    Returns ``None`` if the PDB fails to parse or load.
    """
    pdb_id = pdb_path.stem
    try:
        ctx_lig = session.load_structure(
            pdb_path, mask_ligand=False, max_total_nodes=5000
        )
        ctx_nolig = session.load_structure(
            pdb_path, mask_ligand=True, max_total_nodes=5000
        )
    except Exception as exc:
        logger.warning("UMA: skipping %s (load failed: %s)", pdb_id, exc)
        return None

    if ctx_lig.residue_count != ctx_nolig.residue_count:
        logger.warning(
            "UMA: %s residue count mismatch ligand-on=%d vs ligand-off=%d",
            pdb_id, ctx_lig.residue_count, ctx_nolig.residue_count,
        )
        return None

    try:
        p_lig = _uma_full_distribution(session, ctx_lig, num_batches=num_batches, seed=seed)
        p_nolig = _uma_full_distribution(session, ctx_nolig, num_batches=num_batches, seed=seed)
    except Exception as exc:
        logger.warning("UMA: %s scoring failed: %s", pdb_id, exc)
        return None

    kl = _kl_per_position(p_lig, p_nolig).cpu().numpy()
    native = ctx_lig.native_sequence.cpu().numpy()

    # Distance to ligand: parser returns full y / y_m (every ligand atom in
    # the PDB), and ctx.residue_count Cα coords come from x[:, 1, :].
    parsed = parse_pdb(str(pdb_path))
    x = parsed["X"]
    mask = parsed["mask"].bool()
    residue_coords = x[:, 1, :][mask]
    if residue_coords.shape[0] >= ctx_lig.residue_count:
        residue_coords = residue_coords[: ctx_lig.residue_count]
    y = parsed["Y"]
    y_m = parsed["Y_m"].bool()
    ligand_coords = y[y_m]

    if ligand_coords.shape[0] == 0:
        # No ligand → distance-to-ligand is undefined and KL should be ~0
        # (the ligand-on and ligand-off paths receive identical input).
        # Skip; nothing for the mechanism figure to say.
        logger.info("UMA: skipping %s (no ligand atoms)", pdb_id)
        return None

    distances = residue_ligand_distances(residue_coords, ligand_coords).cpu().numpy()

    rows = []
    for i in range(ctx_lig.residue_count):
        nat = int(native[i])
        if nat == 20:
            continue
        rows.append(
            {
                "pdb_id": pdb_id,
                "model": "uma-inverse-v3",
                "residue_idx": i,
                "native_token": nat,
                "dist_to_ligand": float(distances[i]),
                "kl_shift": float(kl[i]),
                "n_ligand_atoms": int(ligand_coords.shape[0]),
                "num_residues": int(ctx_lig.residue_count),
            }
        )
    return rows


# ─── LigandMPNN ───────────────────────────────────────────────────────────────


def _run_ligandmpnn_score(
    pdb_path: Path,
    ckpt: Path,
    ligandmpnn_dir: Path,
    micromamba_env: str,
    out_dir: Path,
    use_atom_context: int,
    num_batches: int,
    seed: int,
) -> Path:
    """Invoke LigandMPNN's score.py once and return the path to the .pt output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # score.py runs after `cd` into the LigandMPNN install dir, so any relative
    # paths would resolve against that dir (not the project root) and fail. Make
    # the PDB/output/checkpoint paths absolute before interpolating them.
    pdb_path = pdb_path.resolve()
    out_dir = out_dir.resolve()
    ckpt = Path(ckpt).resolve()
    cmd_inner = (
        f"cd '{ligandmpnn_dir}' && "
        f"python score.py "
        f"--model_type ligand_mpnn "
        f"--checkpoint_ligand_mpnn '{ckpt}' "
        f"--pdb_path '{pdb_path}' "
        f"--out_folder '{out_dir}' "
        f"--autoregressive_score 1 "
        f"--use_sequence 1 "
        f"--ligand_mpnn_use_atom_context {use_atom_context} "
        f"--number_of_batches {num_batches} "
        f"--batch_size 1 "
        f"--seed {seed}"
    )
    cmd = [
        "bash", "-c",
        f"eval \"$(micromamba shell hook --shell bash)\" && "
        f"micromamba activate {micromamba_env} && {cmd_inner}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"LigandMPNN score.py failed for {pdb_path.stem} "
            f"(use_atom_context={use_atom_context}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # score.py writes <out_folder>/<pdb_stem><file_ending>.pt — file_ending
    # defaults to empty.
    candidate = out_dir / f"{pdb_path.stem}.pt"
    if not candidate.exists():
        matches = list(out_dir.rglob(f"{pdb_path.stem}*.pt"))
        if not matches:
            raise FileNotFoundError(
                f"LigandMPNN output not found for {pdb_path.stem} under {out_dir}"
            )
        candidate = matches[0]
    return candidate


def _load_lmpnn_probs(score_pt: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Load LigandMPNN score output and return (mean probabilities, native_seq)."""
    blob = torch.load(score_pt, map_location="cpu", weights_only=False)
    log_probs = blob["log_probs"]  # numpy [num_batches, L, 21]
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.cpu().numpy()
    log_probs_t = torch.from_numpy(np.asarray(log_probs)).float()
    if log_probs_t.ndim == 2:
        log_probs_t = log_probs_t.unsqueeze(0)
    probs = log_probs_t.exp().mean(dim=0)  # [L, 21]

    native = blob.get("native_sequence", None)
    if native is None:
        # Fallback: argmax of mean probs is not the same; just return zeros
        native = np.zeros(probs.shape[0], dtype=np.int64)
    elif isinstance(native, torch.Tensor):
        native = native.cpu().numpy()
    else:
        native = np.asarray(native)
    return probs, native


def _eval_ligandmpnn_one_pdb(
    pdb_path: Path,
    ckpt: Path,
    ligandmpnn_dir: Path,
    micromamba_env: str,
    work_dir: Path,
    *,
    num_batches: int,
    seed: int,
) -> list[dict] | None:
    pdb_id = pdb_path.stem
    try:
        score_lig = _run_ligandmpnn_score(
            pdb_path, ckpt, ligandmpnn_dir, micromamba_env,
            work_dir / f"{pdb_id}_ligand_on", use_atom_context=1,
            num_batches=num_batches, seed=seed,
        )
        score_nolig = _run_ligandmpnn_score(
            pdb_path, ckpt, ligandmpnn_dir, micromamba_env,
            work_dir / f"{pdb_id}_ligand_off", use_atom_context=0,
            num_batches=num_batches, seed=seed,
        )
    except Exception as exc:
        logger.warning("LigandMPNN: skipping %s (%s)", pdb_id, exc)
        return None

    probs_lig, native_lig = _load_lmpnn_probs(score_lig)
    probs_nolig, native_nolig = _load_lmpnn_probs(score_nolig)
    if probs_lig.shape != probs_nolig.shape:
        logger.warning("LigandMPNN: %s shape mismatch", pdb_id)
        return None
    if not np.array_equal(native_lig, native_nolig):
        logger.warning("LigandMPNN: %s native sequence mismatch across passes", pdb_id)

    kl = _kl_per_position(probs_lig, probs_nolig).numpy()

    parsed = parse_pdb(str(pdb_path))
    x = parsed["X"]
    mask = parsed["mask"].bool()
    residue_coords = x[:, 1, :][mask]
    if residue_coords.shape[0] >= probs_lig.shape[0]:
        residue_coords = residue_coords[: probs_lig.shape[0]]
    y = parsed["Y"]
    y_m = parsed["Y_m"].bool()
    ligand_coords = y[y_m]
    if ligand_coords.shape[0] == 0:
        logger.info("LigandMPNN: skipping %s (no ligand atoms)", pdb_id)
        return None
    distances = residue_ligand_distances(residue_coords, ligand_coords).cpu().numpy()

    rows = []
    L = probs_lig.shape[0]
    for i in range(L):
        nat = int(native_lig[i]) if i < len(native_lig) else 20
        if nat == 20:
            continue
        rows.append(
            {
                "pdb_id": pdb_id,
                "model": "ligandmpnn",
                "residue_idx": i,
                "native_token": nat,
                "dist_to_ligand": float(distances[i]) if i < len(distances) else float("nan"),
                "kl_shift": float(kl[i]),
                "n_ligand_atoms": int(ligand_coords.shape[0]),
                "num_residues": int(L),
            }
        )
    return rows


# ─── Summary ──────────────────────────────────────────────────────────────────


def _summarize_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, distance-bin) summary: n_residues, n_pdbs, mean KL, std KL."""
    rows = []
    for model_name in sorted(df["model"].unique()):
        sub = df[df["model"] == model_name]
        for label, lo, hi in DISTANCE_BINS:
            in_bin = sub[(sub["dist_to_ligand"] >= lo) & (sub["dist_to_ligand"] < hi)]
            n_res = len(in_bin)
            n_pdbs = in_bin["pdb_id"].nunique() if n_res else 0
            mean_kl = float(in_bin["kl_shift"].mean()) if n_res else float("nan")
            std_kl = float(in_bin["kl_shift"].std()) if n_res > 1 else float("nan")
            median_kl = float(in_bin["kl_shift"].median()) if n_res else float("nan")
            rows.append(
                {
                    "model": model_name,
                    "distance_bin": label,
                    "lo": lo,
                    "hi": None if hi == float("inf") else hi,
                    "n_residues": n_res,
                    "n_pdbs": n_pdbs,
                    "mean_kl": mean_kl,
                    "median_kl": median_kl,
                    "std_kl": std_kl,
                }
            )
    return pd.DataFrame(rows)


# ─── PDB list resolution ──────────────────────────────────────────────────────


def _resolve_pdb_paths(
    pdb_list_path: Path,
    pdb_dirs: list[Path],
    max_residues: int | None,
) -> list[Path]:
    """Resolve PDB IDs / paths to filesystem paths under the configured dirs.

    Each line is either an absolute/relative path to a `.pdb`, or a 4-letter
    PDB ID (looked up in the configured pdb_dirs). Lines starting with `#`
    are skipped.
    """
    raw = [
        s.strip()
        for s in pdb_list_path.read_text().splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]
    paths: list[Path] = []
    for entry in raw:
        as_path = Path(entry)
        if as_path.exists() and as_path.suffix == ".pdb":
            paths.append(as_path)
            continue
        # Treat as PDB ID
        found = None
        for pdb_dir in pdb_dirs:
            cand = resolve_pdb_path(str(pdb_dir), entry)
            if cand is not None:
                found = Path(cand)
                break
        if found is None:
            logger.warning("could not resolve PDB %s in any of %s", entry, pdb_dirs)
            continue
        paths.append(found)

    if max_residues is not None:
        kept: list[Path] = []
        for p in paths:
            try:
                parsed = parse_pdb(str(p))
                n = int(parsed["mask"].sum().item())
            except Exception:
                logger.debug("could not parse %s for size filter", p)
                continue
            if n <= max_residues:
                kept.append(p)
            else:
                logger.info(
                    "filtering %s (%d > max_residues=%d) — out of v3 crop budget",
                    p.stem, n, max_residues,
                )
        paths = kept
    return paths


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uma-ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "uma-inverse-v3.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "old_configs" / "config_v3.yaml",
        help="Hydra config matching the ckpt's training-time data flags.",
    )
    parser.add_argument(
        "--pdb-list",
        type=Path,
        required=True,
        help="Newline-delimited file of PDB IDs or paths.",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        action="append",
        default=None,
        help="One or more directories to look up PDB IDs in (repeatable). "
        "Defaults to data/raw/pdb_archive plus its test_metal/ and "
        "test_small_molecule/ subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "distal_kl",
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=359,
        help=(
            "Filter out PDBs above this residue count to avoid v3 cropping "
            "(max_total_nodes=384 minus 25 ligand atoms). Set to 0 to disable."
        ),
    )
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-uma", action="store_true")
    parser.add_argument("--skip-ligandmpnn", action="store_true")
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
    args = parser.parse_args()

    if args.pdb_dir is None:
        archive = PROJECT_ROOT / "data" / "raw" / "pdb_archive"
        args.pdb_dir = [
            archive,
            archive / "test_metal",
            archive / "test_small_molecule",
        ]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    max_residues = args.max_residues if args.max_residues > 0 else None
    pdb_paths = _resolve_pdb_paths(args.pdb_list, args.pdb_dir, max_residues)
    logger.info("resolved %d PDBs", len(pdb_paths))
    if not pdb_paths:
        logger.error("no PDBs resolved; check --pdb-list and --pdb-dir")
        return

    all_rows: list[dict] = []

    if not args.skip_uma:
        if not args.uma_ckpt.exists():
            logger.error("UMA ckpt not found: %s", args.uma_ckpt)
        else:
            logger.info("loading UMA-Inverse v3 from %s", args.uma_ckpt)
            session = InferenceSession.from_checkpoint(
                config_path=args.config, checkpoint=args.uma_ckpt
            )
            for idx, pdb_path in enumerate(pdb_paths, 1):
                logger.info("UMA %d/%d: %s", idx, len(pdb_paths), pdb_path.stem)
                rows = _eval_uma_one_pdb(
                    session, pdb_path,
                    num_batches=args.num_batches, seed=args.seed,
                )
                if rows:
                    all_rows.extend(rows)

    if not args.skip_ligandmpnn:
        if not args.ligandmpnn_ckpt.exists():
            logger.error("LigandMPNN ckpt not found: %s", args.ligandmpnn_ckpt)
        else:
            with tempfile.TemporaryDirectory(prefix="lmpnn_kl_") as tmpdir:
                work_dir = Path(tmpdir)
                for idx, pdb_path in enumerate(pdb_paths, 1):
                    logger.info(
                        "LigandMPNN %d/%d: %s", idx, len(pdb_paths), pdb_path.stem
                    )
                    rows = _eval_ligandmpnn_one_pdb(
                        pdb_path,
                        args.ligandmpnn_ckpt,
                        args.ligandmpnn_dir,
                        args.micromamba_env,
                        work_dir,
                        num_batches=args.num_batches,
                        seed=args.seed,
                    )
                    if rows:
                        all_rows.extend(rows)

    if not all_rows:
        logger.error("no rows produced; aborting")
        return

    df = pd.DataFrame(all_rows)
    parquet_path = args.out_dir / "per_position.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("wrote %s (%d rows)", parquet_path, len(df))

    summary_df = _summarize_by_bin(df)
    summary_path = args.out_dir / "distal_kl_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("wrote %s (%d rows)", summary_path, len(summary_df))

    manifest = {
        "uma_ckpt": str(args.uma_ckpt),
        "ligandmpnn_ckpt": str(args.ligandmpnn_ckpt),
        "num_batches": args.num_batches,
        "seed": args.seed,
        "max_residues": max_residues,
        "n_pdbs_input": len(pdb_paths),
        "n_pdbs_with_uma_rows": int(df[df["model"] == "uma-inverse-v3"]["pdb_id"].nunique()),
        "n_pdbs_with_ligandmpnn_rows": int(
            df[df["model"] == "ligandmpnn"]["pdb_id"].nunique()
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
