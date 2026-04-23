"""Typer subcommand: ``uma-inverse benchmark``.

Wires the full evaluation + sweep + report pipeline behind a single
command. Designed so *one* invocation produces every number the paper
needs — no post-processing required for the standard tables.

Implementation lives in a separate Typer app so the inference CLI can
register it via ``add_typer`` without importing the heavyweight benchmark
dependencies until the command is actually invoked.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from omegaconf import OmegaConf

from src.benchmarks.evaluation import evaluate_validation_set
from src.benchmarks.report import write_report
from src.benchmarks.sweeps import (
    format_timing,
    run_ligand_ablation,
    run_temperature_sweep,
)
from src.inference.output import build_manifest
from src.inference.session import InferenceSession

logger = logging.getLogger(__name__)

bench_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Run the full benchmark pipeline against a trained checkpoint. "
        "Produces every table and figure the paper needs in one directory."
    ),
)


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_temperatures(raw: str) -> list[float]:
    try:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"invalid --temperatures: {raw!r}") from exc


@bench_app.callback(invoke_without_command=True)
def benchmark(
    # ── Inputs ────────────────────────────────────────────────────────────────
    ckpt: Path = typer.Option(
        ..., "--ckpt", exists=True, dir_okay=False, readable=True,
        help="Trained checkpoint (.ckpt)."
    ),
    val_json: Path = typer.Option(
        ..., "--val-json", exists=True, dir_okay=False, readable=True,
        help="LigandMPNN-style JSON list of validation PDB ids.",
    ),
    pdb_dir: Path = typer.Option(
        ..., "--pdb-dir", exists=True, file_okay=False,
        help="Directory tree holding the parsed PDB files.",
    ),
    out_dir: Path = typer.Option(
        Path("outputs/benchmark"), "--out-dir", file_okay=False,
        help="Output directory (a run-name subdirectory is created inside).",
    ),
    config_path: Path = typer.Option(
        Path("configs/config.yaml"), "--config",
        exists=True, dir_okay=False, readable=True,
        help="Hydra config YAML matching the checkpoint.",
    ),
    # ── Scope ─────────────────────────────────────────────────────────────────
    n_pdbs: int | None = typer.Option(
        500, "--n-pdbs",
        help="Cap on number of val PDBs to evaluate (random subsample). --all for the full split."
    ),
    run_all: bool = typer.Option(
        False, "--all",
        help="Evaluate every resolvable PDB in --val-json (supersedes --n-pdbs)."
    ),
    run_name: str | None = typer.Option(None, "--run-name"),
    seed: int = typer.Option(0, "--seed", help="Subsample + RNG seed for reproducibility."),
    max_total_nodes: int | None = typer.Option(
        None, "--max-total-nodes",
        help="Override the configured residue-crop budget. "
             "Benchmarks should use a large value to avoid silent cropping."
    ),
    # ── Sweeps ────────────────────────────────────────────────────────────────
    skip_ablation: bool = typer.Option(False, "--skip-ablation",
        help="Skip the ligand-context ablation (halves runtime)."),
    skip_temperature: bool = typer.Option(False, "--skip-temperature",
        help="Skip the temperature/diversity sweep."),
    temperatures: str = typer.Option("0.0,0.1,0.2,0.5,1.0", "--temperatures",
        help="Comma-separated list of sampling temperatures."),
    samples_per_pdb: int = typer.Option(3, "--samples-per-pdb", min=2,
        help="Samples at each T per PDB (for diversity measurements)."),
    # ── Environment ──────────────────────────────────────────────────────────
    device: str = typer.Option("auto", "--device", help="'cuda', 'cpu', or 'auto'."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Run validation evaluation + ligand ablation + temperature sweep."""
    _configure_logging(verbose)

    resolved_n_pdbs: int | None = None if run_all else n_pdbs
    if run_all and n_pdbs is not None:
        logger.info("--all supplied; ignoring --n-pdbs=%s", n_pdbs)

    # ── Output directory ─────────────────────────────────────────────────────
    resolved_run_name = run_name or f"{ckpt.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = out_dir / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Session ─────────────────────────────────────────────────────────────
    logger.info("loading checkpoint: %s", ckpt)
    session = InferenceSession.from_checkpoint(
        config_path=config_path, checkpoint=ckpt, device=device,
    )

    # ── Manifest (pre-run; updated at the end) ───────────────────────────────
    config_snapshot = OmegaConf.to_container(session.config, resolve=True)
    manifest = build_manifest(
        run_name=resolved_run_name,
        command=" ".join(sys.argv),
        checkpoint_path=str(ckpt),
        config_path=str(config_path),
        config_snapshot=config_snapshot,  # type: ignore[arg-type]
        seed=seed,
        temperature=0.0,
        top_p=None,
        decoding_order="fixed",
        num_pdbs=resolved_n_pdbs or 0,
        num_samples_per_pdb=samples_per_pdb,
        extras={
            "val_json": str(val_json),
            "pdb_dir": str(pdb_dir),
            "max_total_nodes": max_total_nodes,
            "skip_ablation": skip_ablation,
            "skip_temperature": skip_temperature,
            "temperatures": temperatures,
        },
    )
    manifest.write(run_dir / "run_manifest.json")

    def progress(idx: int, total: int, label: str) -> None:
        if idx % max(1, total // 20) == 0:
            logger.info("  progress: %d/%d — %s", idx, total, label)

    # ── Pass 1: teacher-forced evaluation ────────────────────────────────────
    logger.info("evaluation: running validation set")
    t_eval = time.perf_counter()
    evaluations = evaluate_validation_set(
        session=session,
        val_json=val_json,
        pdb_dir=pdb_dir,
        n_pdbs=resolved_n_pdbs,
        mask_ligand=False,
        max_total_nodes=max_total_nodes,
        seed=seed,
        progress_callback=progress,
    )
    logger.info("evaluation finished in %s (%d PDBs)",
                format_timing(time.perf_counter() - t_eval), len(evaluations))

    # ── Pass 2: ligand-context ablation ─────────────────────────────────────
    ablation_rows = None
    if not skip_ablation:
        logger.info("ablation: running ligand-context sweep")
        t_ab = time.perf_counter()
        ablation_rows = run_ligand_ablation(
            session=session,
            val_json=val_json,
            pdb_dir=pdb_dir,
            n_pdbs=resolved_n_pdbs,
            max_total_nodes=max_total_nodes,
            seed=seed,
            progress_callback=progress,
        )
        logger.info("ablation finished in %s (%d paired PDBs)",
                    format_timing(time.perf_counter() - t_ab), len(ablation_rows))

    # ── Pass 3: temperature sweep ────────────────────────────────────────────
    temperature_rows = None
    if not skip_temperature:
        logger.info("temperature sweep")
        t_sweep = time.perf_counter()
        temperature_rows = run_temperature_sweep(
            session=session,
            val_json=val_json,
            pdb_dir=pdb_dir,
            temperatures=_parse_temperatures(temperatures),
            num_samples_per_pdb=samples_per_pdb,
            n_pdbs=resolved_n_pdbs,
            max_total_nodes=max_total_nodes,
            seed=seed,
            progress_callback=progress,
        )
        logger.info("sweep finished in %s", format_timing(time.perf_counter() - t_sweep))

    # ── Report ───────────────────────────────────────────────────────────────
    logger.info("aggregating report")
    write_report(
        out_dir=run_dir,
        evaluations=evaluations,
        ablation_rows=ablation_rows,
        temperature_rows=temperature_rows,
        run_metadata={
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "config_path": str(config_path),
            "git_hash": manifest.git_hash,
            "start_timestamp": manifest.start_timestamp,
        },
    )

    manifest.stop_timestamp = datetime.now().isoformat()
    manifest.write(run_dir / "run_manifest.json")

    typer.echo(f"done: {run_dir}")
    typer.echo(f"      read {run_dir / 'summary.md'} for headline numbers")
