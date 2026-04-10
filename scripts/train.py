"""UMA-Inverse training entry point (Hydra CLI).

Run via Makefile:
    make train          # sbatch full curriculum
    make pilot          # srun 1-batch sanity check

Direct invocation (for debugging only — never on cluster without srun):
    uv run python scripts/train.py
    uv run python scripts/train.py ++data.max_total_nodes=64
"""
import logging
import os
import subprocess
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.datamodule import UMAInverseDataModule
from src.training.lightning_module import UMAInverseLightningModule

logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_run_metadata(cfg: DictConfig, run_dir: str) -> None:
    """Persist config + git hash alongside the run for full reproducibility."""
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
    git_hash = _get_git_hash()
    with open(os.path.join(run_dir, "git_hash.txt"), "w") as f:
        f.write(git_hash + "\n")
    logger.info("Run metadata saved to %s (git: %s)", run_dir, git_hash[:8])


def _build_loggers(cfg: DictConfig, run_name: str):
    """Always return a CSVLogger; optionally add W&B when enabled."""
    loggers = [CSVLogger(save_dir=os.path.join(PROJECT_ROOT, "logs", "csv"), name=run_name)]

    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            loggers.append(
                WandbLogger(
                    project=cfg.get("project_name", "UMA-Inverse"),
                    name=run_name,
                    mode=wandb_cfg.get("mode", "offline"),
                    save_dir=os.path.join(PROJECT_ROOT, "logs", "wandb"),
                )
            )
            logger.info("W&B logger enabled (mode=%s)", wandb_cfg.get("mode", "offline"))
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logger")

    return loggers


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)
    torch.set_float32_matmul_precision("high")

    run_name = cfg.get("run_name", "pairmixer-run")
    run_dir = os.path.join(PROJECT_ROOT, "logs", "runs", run_name)
    _save_run_metadata(cfg, run_dir)

    # ── Paths ──────────────────────────────────────────────────────────────────
    train_json = hydra.utils.to_absolute_path(cfg.paths.train_json)
    valid_json = hydra.utils.to_absolute_path(cfg.paths.valid_json)
    pdb_dir    = hydra.utils.to_absolute_path(cfg.paths.pdb_dir)

    if not os.path.exists(train_json):
        raise FileNotFoundError(f"train_json not found: {train_json}")
    if not os.path.exists(valid_json):
        raise FileNotFoundError(f"valid_json not found: {valid_json}")

    max_total_nodes = int(cfg.data.get("max_total_nodes", 384))

    # ── Data ───────────────────────────────────────────────────────────────────
    datamodule = UMAInverseDataModule(
        train_json=train_json,
        valid_json=valid_json,
        pdb_dir=pdb_dir,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        ligand_context_atoms=int(cfg.data.ligand_context_atoms),
        cutoff_for_score=float(cfg.data.cutoff_for_score),
        max_total_nodes=max_total_nodes,
    )

    # ── LR schedule parameters ─────────────────────────────────────────────────
    warmup_steps = int(cfg.training.get("warmup_steps", 500))
    T_max        = int(cfg.training.get("T_max", 50_000))

    # ── Model ──────────────────────────────────────────────────────────────────
    model = UMAInverseLightningModule(
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        warmup_steps=warmup_steps,
        T_max=T_max,
        compile_model=bool(cfg.training.compile_model),
    )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="uma-inverse-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/loss",
            patience=int(cfg.training.get("early_stop_patience", 10)),
            mode="min",
        ),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    # ── Trainer ────────────────────────────────────────────────────────────────
    accelerator = str(cfg.training.accelerator)
    precision   = str(cfg.training.precision)
    if accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU requested but not available — falling back to CPU/fp32")
        accelerator = "cpu"
        precision   = "32-true"

    max_epochs = int(cfg.training.epochs)
    if cfg.get("trainer") and cfg.trainer.get("max_epochs"):
        max_epochs = int(cfg.trainer.max_epochs)

    os.makedirs(os.path.join(PROJECT_ROOT, "logs", "wandb"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs", "SLURM_out"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs", "SLURM_err"), exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=int(cfg.training.devices),
        precision=precision,
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        callbacks=callbacks,
        logger=_build_loggers(cfg, run_name),
        default_root_dir=".",
        num_sanity_val_steps=2,
    )

    ckpt_path = None
    if cfg.get("trainer") and cfg.trainer.get("resume_from_checkpoint"):
        ckpt_path = cfg.trainer.resume_from_checkpoint
        if not os.path.exists(ckpt_path):
            logger.warning("resume_from_checkpoint path not found: %s", ckpt_path)
            ckpt_path = None

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
