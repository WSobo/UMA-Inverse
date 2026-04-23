"""``uma-inverse`` command-line entry point.

Thin layer over the library modules in :mod:`src.inference`. Each
subcommand:

1. Parses CLI flags with Typer.
2. Builds a :class:`DesignConstraints` (or scoring spec) from them.
3. Opens one :class:`InferenceSession`, loads structures, decodes/scores.
4. Routes produced records through :mod:`src.inference.output` for
   FASTA / JSON / npz serialisation.

Subcommands
-----------
``uma-inverse design`` — sample new sequences for one or many PDBs.
``uma-inverse score`` — log-likelihood scoring for the native or a user-
    supplied sequence.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import typer
from omegaconf import OmegaConf

from src.inference.batch import (
    BatchEntry,
    append_done,
    filter_pending,
    load_batch_spec,
    merge_constraint_kwargs,
)
from src.inference.constraints import DesignConstraints, as_token_ids
from src.inference.decoding import autoregressive_design, score_sequence
from src.inference.output import (
    build_manifest,
    build_ranked_rows,
    ligand_neighbour_mask_from_ctx,
    write_per_residue_confidence,
    write_probs_npz,
    write_ranked_csv,
    write_samples_fasta,
)
from src.inference.session import InferenceSession

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "UMA-Inverse command-line interface. "
        "Use 'design' to generate sequences, 'score' to compute log-likelihoods, "
        "or 'benchmark' to run the full paper evaluation pipeline."
    ),
)

# Benchmark subcommand is registered via add_typer so its heavier imports
# (matplotlib, pandas, pyarrow) are only loaded when actually invoked.
from src.benchmarks.cli import bench_app as _bench_app  # noqa: E402

app.add_typer(
    _bench_app,
    name="benchmark",
    help="Run full validation benchmark suite (recovery, calibration, ablation, sweep).",
)

logger = logging.getLogger(__name__)


@app.callback()
def _cli_root() -> None:
    """UMA-Inverse inference CLI.

    A callback is registered so Typer always dispatches to a named
    subcommand rather than promoting a single ``@app.command`` to root.
    """
    return None


# ─── Shared helpers ───────────────────────────────────────────────────────────


def _configure_logging(verbose: int) -> None:
    """Map ``-v``/``-q`` flag counts to standard log levels."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose < 0:
        level = logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class DesignFlags:
    """CLI surface collected into one record for forwarding between helpers."""

    num_samples: int
    batch_size: int
    temperature: float
    top_p: float | None
    seed: int | None
    decoding_order: str
    suffix: str
    save_probs: bool
    write_ranked: bool
    include_native: bool


def _resolve_run_dir(out_dir: Path, run_name: str | None, stem: str) -> Path:
    resolved = run_name or f"{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = out_dir / resolved
    path.mkdir(parents=True, exist_ok=True)
    return path


def _design_one_pdb(
    *,
    session: InferenceSession,
    pdb_path: Path,
    constraint_kwargs: dict[str, Any],
    flags: DesignFlags,
    run_dir: Path,
    sample_index_base: int = 0,
    append_ranked: bool = False,
) -> int:
    """Design samples for a single PDB and write outputs.

    Returns the number of samples written. Exists as a standalone helper so
    both ``design`` (one PDB) and ``batch`` (many PDBs, this function in a
    loop) share the same logic.
    """
    constraints = DesignConstraints.from_cli(**constraint_kwargs)
    logger.info("loading structure: %s", pdb_path)
    ctx = session.load_structure(
        pdb_path=pdb_path,
        parse_chains=constraints.parse_chains,
        include_zero_occupancy=constraints.include_zero_occupancy,
        ligand_cutoff=constraints.ligand_cutoff,
        mask_ligand=constraints.mask_ligand,
        max_total_nodes=constraints.max_total_nodes,
    )
    resolved = constraints.resolve(ctx)
    n_designable = int(resolved.designable_mask.sum().item())
    n_fixed = int(resolved.fixed_mask.sum().item())
    logger.info(
        "%s: %d residues, %d designable, %d fixed",
        pdb_path.name, ctx.residue_count, n_designable, n_fixed,
    )

    start = time.perf_counter()
    samples = autoregressive_design(
        session=session,
        ctx=ctx,
        constraints=resolved,
        num_samples=flags.num_samples,
        batch_size=flags.batch_size,
        temperature=flags.temperature,
        top_p=flags.top_p,
        seed=(flags.seed + sample_index_base) if flags.seed is not None else None,
        decoding_order=flags.decoding_order,  # type: ignore[arg-type]
    )
    elapsed = time.perf_counter() - start
    logger.info(
        "%s: designed %d samples in %.2fs (%.2fs / sample)",
        pdb_path.name, len(samples), elapsed, elapsed / max(len(samples), 1),
    )

    pdb_id = pdb_path.stem
    base = pdb_id + flags.suffix

    fasta_dir = run_dir / "fastas"
    conf_dir = run_dir / "confidences"
    probs_dir = run_dir / "probs"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    if flags.save_probs:
        probs_dir.mkdir(parents=True, exist_ok=True)

    ligand_mask = ligand_neighbour_mask_from_ctx(ctx, cutoff=0.0)

    write_samples_fasta(
        fasta_dir / f"{base}.fa",
        pdb_id=pdb_id,
        ctx=ctx,
        samples=samples,
        designable_mask=resolved.designable_mask.cpu(),
        ligand_neighbour_mask=ligand_mask.cpu(),
        include_native=flags.include_native,
    )
    write_per_residue_confidence(
        conf_dir / f"{base}.json",
        pdb_id=pdb_id, ctx=ctx, samples=samples,
        designable_mask=resolved.designable_mask.cpu(),
    )
    if flags.save_probs:
        write_probs_npz(
            probs_dir / f"{base}.npz",
            pdb_id=pdb_id, ctx=ctx, samples=samples,
        )
    if flags.write_ranked and flags.num_samples > 1:
        rows = build_ranked_rows(
            pdb_id=pdb_id,
            ctx=ctx, samples=samples,
            designable_mask=resolved.designable_mask.cpu(),
            ligand_neighbour_mask=ligand_mask.cpu(),
        )
        write_ranked_csv(run_dir / "ranked.csv", rows)

    return len(samples)


# ─── 'design' subcommand ──────────────────────────────────────────────────────


@app.command(
    "design",
    help="Generate sequences for one or many PDBs under optional constraints.",
    no_args_is_help=True,
)
def design(
    # ── Input selection (mutually exclusive: --pdb xor --pdb-list) ──────────
    pdb: Path | None = typer.Option(
        None, "--pdb", exists=True, dir_okay=False, readable=True,
        help="Single PDB to design. Mutually exclusive with --pdb-list.",
    ),
    pdb_list: Path | None = typer.Option(
        None, "--pdb-list", exists=True, dir_okay=False, readable=True,
        help="Batch JSON: {pdb_path: {overrides...}} or {pdb_path: {}}.",
    ),
    ckpt: Path = typer.Option(
        ..., "--ckpt", exists=True, dir_okay=False, readable=True,
        help="Checkpoint file (.ckpt) produced by training.",
    ),
    out_dir: Path = typer.Option(
        Path("outputs"), "--out-dir", file_okay=False,
        help="Output directory. A run-name subdirectory is created here.",
    ),
    config_path: Path = typer.Option(
        Path("configs/config.yaml"), "--config",
        exists=True, dir_okay=False, readable=True,
        help="Hydra config YAML matching the checkpoint's model settings.",
    ),
    # ── Sampling ─────────────────────────────────────────────────────────────
    num_samples: int = typer.Option(1, "--num-samples", min=1,
        help="Number of sequences to sample per PDB."),
    batch_size: int = typer.Option(1, "--batch-size", min=1,
        help="Samples decoded in parallel per forward pass (GPU memory dial)."),
    temperature: float = typer.Option(0.1, "--temperature", min=0.0,
        help="Sampling temperature. 0.0 = argmax."),
    top_p: float | None = typer.Option(None, "--top-p",
        help="Nucleus (top-p) threshold in (0, 1]. Omit to disable."),
    seed: int | None = typer.Option(None, "--seed",
        help="Base RNG seed (sample i uses seed+i). Random if omitted."),
    decoding_order: str = typer.Option("random", "--decoding-order",
        help="'random' (matches LigandMPNN) or 'left-to-right' for debugging."),
    # ── Residue selection ────────────────────────────────────────────────────
    fix: str | None = typer.Option(None, "--fix",
        help="Residues to preserve (e.g. 'A1 A2 B42C' or 'A1,A2')."),
    redesign: str | None = typer.Option(None, "--redesign",
        help="Residues to redesign (complement is held native)."),
    design_chains: str | None = typer.Option(None, "--design-chains",
        help="Chains to redesign, comma-separated (e.g. 'A,B')."),
    parse_chains: str | None = typer.Option(None, "--parse-chains",
        help="Only parse these chains into the structure."),
    # ── Constraints ──────────────────────────────────────────────────────────
    bias: str | None = typer.Option(None, "--bias",
        help="Global AA bias, e.g. 'W:3.0,A:-1.0'."),
    bias_file: Path | None = typer.Option(None, "--bias-file",
        exists=True, dir_okay=False, readable=True,
        help='Per-residue bias JSON: {"A23": {"W": 3.0}, ...}.'),
    omit: str | None = typer.Option(None, "--omit",
        help="Global AA omit, e.g. 'CDFG' or 'C,D,F,G'."),
    omit_file: Path | None = typer.Option(None, "--omit-file",
        exists=True, dir_okay=False, readable=True,
        help='Per-residue omit JSON: {"A23": "CDFG", ...}.'),
    tie: str | None = typer.Option(None, "--tie",
        help="Tie groups, e.g. 'A1,A10|B5,B15' (groups sep by |)."),
    tie_weights: str | None = typer.Option(None, "--tie-weights",
        help="Weights aligned to --tie. Defaults to equal per group."),
    # ── Structure / parser flags ─────────────────────────────────────────────
    mask_ligand: bool = typer.Option(False, "--mask-ligand",
        help="Zero ligand atom features (ablation)."),
    ligand_cutoff: float | None = typer.Option(None, "--ligand-cutoff", min=0.0,
        help="Å cutoff for ligand-proximal scoring. Default: config (8.0)."),
    include_zero_occupancy: bool = typer.Option(False, "--include-zero-occupancy",
        help="Parse atoms with occupancy=0 (default: skip)."),
    max_total_nodes: int | None = typer.Option(None, "--max-total-nodes", min=1,
        help="Cap residues+ligand atoms after cropping. Default: config value."),
    # ── Output ───────────────────────────────────────────────────────────────
    run_name: str | None = typer.Option(None, "--run-name",
        help="Subdirectory under --out-dir. Default: <stem>-<timestamp>."),
    suffix: str = typer.Option("", "--suffix",
        help="File ending appended to output basenames (e.g. '_v2')."),
    save_probs: bool = typer.Option(False, "--save-probs",
        help="Also dump the full [N, L, 21] probability tensor as .npz."),
    write_ranked: bool = typer.Option(True, "--ranked/--no-ranked",
        help="Write dedup'd ranked.csv across produced samples."),
    include_native: bool = typer.Option(True, "--include-native/--no-native",
        help="Include the parsed native sequence as the first FASTA record."),
    # ── Batch control ────────────────────────────────────────────────────────
    resume: bool = typer.Option(False, "--resume",
        help="Skip PDBs already recorded in .done.txt (batch mode only)."),
    # ── Environment ──────────────────────────────────────────────────────────
    device: str = typer.Option("auto", "--device",
        help="'cuda', 'cpu', or 'auto'."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True,
        help="-v for INFO, -vv for DEBUG."),
) -> None:
    """Generate designs for a single PDB or a batch of PDBs."""
    _configure_logging(verbose)

    if (pdb is None) == (pdb_list is None):
        raise typer.BadParameter(
            "provide exactly one of --pdb or --pdb-list"
        )

    # ── Resolve entries ─────────────────────────────────────────────────────
    if pdb_list is not None:
        entries = load_batch_spec(pdb_list)
        if not entries:
            raise typer.BadParameter(f"batch spec is empty: {pdb_list}")
        run_stem = pdb_list.stem
    else:
        entries = [BatchEntry(pdb_path=pdb)]  # type: ignore[arg-type]
        run_stem = pdb.stem  # type: ignore[union-attr]

    run_dir = _resolve_run_dir(out_dir, run_name, run_stem)

    # ── Session ─────────────────────────────────────────────────────────────
    logger.info("loading checkpoint: %s", ckpt)
    session = InferenceSession.from_checkpoint(
        config_path=config_path, checkpoint=ckpt, device=device,
    )

    # ── Filter for resume ───────────────────────────────────────────────────
    done_path = run_dir / ".done.txt"
    pending = filter_pending(entries, done_path=done_path, resume=resume)
    if not pending:
        typer.echo(f"nothing to do: all {len(entries)} PDBs already in {done_path}")
        raise typer.Exit(code=0)

    # ── CLI defaults for constraint kwargs ──────────────────────────────────
    cli_constraint_defaults: dict[str, Any] = {
        "fix": fix,
        "redesign": redesign,
        "design_chains": design_chains,
        "parse_chains": parse_chains,
        "bias": bias,
        "bias_file": bias_file,
        "omit": omit,
        "omit_file": omit_file,
        "tie": tie,
        "tie_weights": tie_weights,
        "mask_ligand": mask_ligand,
        "include_zero_occupancy": include_zero_occupancy,
        "ligand_cutoff": ligand_cutoff,
        "max_total_nodes": max_total_nodes,
    }

    flags = DesignFlags(
        num_samples=num_samples,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        decoding_order=decoding_order,
        suffix=suffix,
        save_probs=save_probs,
        write_ranked=write_ranked,
        include_native=include_native,
    )

    # ── Manifest (written once at start so crashes leave a trace) ───────────
    config_snapshot = OmegaConf.to_container(session.config, resolve=True)
    manifest = build_manifest(
        run_name=run_dir.name,
        command=" ".join(sys.argv),
        checkpoint_path=str(ckpt),
        config_path=str(config_path),
        config_snapshot=config_snapshot,  # type: ignore[arg-type]
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        decoding_order=decoding_order,
        num_pdbs=len(entries),
        num_samples_per_pdb=num_samples,
        extras={
            "pdb_list": str(pdb_list) if pdb_list else None,
            "pdb": str(pdb) if pdb else None,
            "batch_mode": pdb_list is not None,
            "resume": resume,
            "cli_constraint_defaults": cli_constraint_defaults,
        },
    )
    manifest.write(run_dir / "run_manifest.json")

    # ── Process each PDB ────────────────────────────────────────────────────
    total_samples = 0
    for idx, entry in enumerate(pending):
        if not entry.pdb_path.exists():
            logger.error("skipping missing PDB: %s", entry.pdb_path)
            continue
        kwargs = merge_constraint_kwargs(cli_constraint_defaults, entry.overrides)
        try:
            n = _design_one_pdb(
                session=session,
                pdb_path=entry.pdb_path,
                constraint_kwargs=kwargs,
                flags=flags,
                run_dir=run_dir,
                sample_index_base=idx * num_samples,  # disjoint seed space per PDB
            )
            total_samples += n
            append_done(done_path, str(entry.pdb_path))
        except Exception:
            logger.exception("failure on %s — continuing", entry.pdb_path)

    manifest.stop_timestamp = datetime.now().isoformat()
    manifest.write(run_dir / "run_manifest.json")

    typer.echo(
        f"done: {run_dir} — {total_samples} samples across {len(pending)} PDB(s)"
    )


# ─── 'score' subcommand ───────────────────────────────────────────────────────


@app.command(
    "score",
    help="Compute per-residue log-likelihoods under the trained model.",
    no_args_is_help=True,
)
def score(
    pdb: Path = typer.Option(
        ..., "--pdb", exists=True, dir_okay=False, readable=True,
        help="PDB file to score against.",
    ),
    ckpt: Path = typer.Option(
        ..., "--ckpt", exists=True, dir_okay=False, readable=True,
        help="Checkpoint file.",
    ),
    config_path: Path = typer.Option(
        Path("configs/config.yaml"), "--config",
        exists=True, dir_okay=False, readable=True,
    ),
    mode: str = typer.Option("autoregressive", "--mode",
        help="'autoregressive' (default) or 'single-aa'."),
    use_sequence: bool = typer.Option(True, "--use-sequence/--no-use-sequence",
        help="Feed the sequence to the AR context (True) or mask as all-X (False)."),
    sequence: str | None = typer.Option(None, "--sequence",
        help="Alternative sequence to score (single-letter AA codes). "
             "Must match the structure's parsed residue count. Default: native."),
    num_batches: int = typer.Option(10, "--num-batches", min=1,
        help="Random decoding orders averaged (AR mode only)."),
    seed: int | None = typer.Option(None, "--seed",
        help="Base seed for random decoding orders."),
    out_dir: Path = typer.Option(
        Path("outputs"), "--out-dir", file_okay=False,
        help="Output directory. Writes scores_<pdb>.csv and scores_<pdb>.json.",
    ),
    run_name: str | None = typer.Option(None, "--run-name"),
    mask_ligand: bool = typer.Option(False, "--mask-ligand"),
    parse_chains: str | None = typer.Option(None, "--parse-chains"),
    max_total_nodes: int | None = typer.Option(None, "--max-total-nodes", min=1),
    ligand_cutoff: float | None = typer.Option(None, "--ligand-cutoff", min=0.0),
    include_zero_occupancy: bool = typer.Option(False, "--include-zero-occupancy"),
    device: str = typer.Option("auto", "--device"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Score the native or a user-supplied sequence position-by-position."""
    _configure_logging(verbose)

    if mode not in ("autoregressive", "single-aa"):
        raise typer.BadParameter(f"mode must be 'autoregressive' or 'single-aa'; got {mode!r}")

    run_dir = _resolve_run_dir(out_dir, run_name, f"{pdb.stem}-score")

    session = InferenceSession.from_checkpoint(
        config_path=config_path, checkpoint=ckpt, device=device,
    )
    parse_chain_list = (
        [c.strip() for c in parse_chains.split(",") if c.strip()] if parse_chains else None
    )
    ctx = session.load_structure(
        pdb_path=pdb,
        parse_chains=parse_chain_list,
        include_zero_occupancy=include_zero_occupancy,
        ligand_cutoff=ligand_cutoff,
        mask_ligand=mask_ligand,
        max_total_nodes=max_total_nodes,
    )

    scored_sequence: torch.Tensor | None = None
    if sequence is not None:
        letters = sequence.strip().upper().replace(" ", "")
        if len(letters) != ctx.residue_count:
            raise typer.BadParameter(
                f"--sequence length {len(letters)} does not match residue count {ctx.residue_count}"
            )
        scored_sequence = torch.tensor(as_token_ids(letters), dtype=torch.long)

    start = time.perf_counter()
    result = score_sequence(
        session=session,
        ctx=ctx,
        sequence=scored_sequence,
        mode=mode,  # type: ignore[arg-type]
        use_sequence=use_sequence,
        num_batches=num_batches,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    logger.info("scoring complete in %.2fs", elapsed)

    # ── Persist ─────────────────────────────────────────────────────────────
    csv_path = run_dir / f"scores_{pdb.stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["position", "residue_id", "aa", "log_prob", "prob"])
        import math as _math
        for i, rid in enumerate(ctx.residue_ids):
            lp = float(result.log_probs[i].item())
            tok = int(result.sequence[i].item())
            from src.utils.io import ID_TO_AA as _ID
            writer.writerow([i, rid, _ID.get(tok, "X"), f"{lp:.6f}", f"{_math.exp(lp):.6f}"])

    summary = {
        "pdb_id": pdb.stem,
        "pdb_path": str(pdb),
        "mode": result.mode,
        "use_sequence": result.use_sequence,
        "num_batches": result.num_batches,
        "mean_log_prob": result.mean_log_prob(),
        "sum_log_prob": float(result.log_probs.sum().item()),
        "residue_count": ctx.residue_count,
    }
    (run_dir / f"scores_{pdb.stem}.json").write_text(json.dumps(summary, indent=2))

    # Manifest for reproducibility parity with design
    config_snapshot = OmegaConf.to_container(session.config, resolve=True)
    manifest = build_manifest(
        run_name=run_dir.name,
        command=" ".join(sys.argv),
        checkpoint_path=str(ckpt),
        config_path=str(config_path),
        config_snapshot=config_snapshot,  # type: ignore[arg-type]
        seed=seed,
        temperature=0.0,
        top_p=None,
        decoding_order="random" if mode == "autoregressive" else "fixed",
        num_pdbs=1,
        num_samples_per_pdb=num_batches,
        extras={
            "pdb": str(pdb),
            "mode": mode,
            "use_sequence": use_sequence,
            "custom_sequence": bool(sequence),
            "summary": summary,
        },
    )
    manifest.stop_timestamp = datetime.now().isoformat()
    manifest.write(run_dir / "run_manifest.json")

    typer.echo(
        f"done: mean log-prob {summary['mean_log_prob']:.4f} across {ctx.residue_count} residues"
    )
    typer.echo(f"outputs: {run_dir}")


# ─── Script entrypoint ────────────────────────────────────────────────────────


def main() -> None:
    """Module entrypoint (exposed as console script ``uma-inverse``)."""
    app()


if __name__ == "__main__":
    main()
