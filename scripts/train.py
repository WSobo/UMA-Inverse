import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.datamodule import UMAInverseDataModule
from src.training.lightning_module import UMAInverseLightningModule


def _build_logger(cfg: DictConfig):
    """Return a WandB logger when enabled, else None (CSV fallback)."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    from pytorch_lightning.loggers import WandbLogger  # lazy import

    return WandbLogger(
        project=cfg.get("project_name", "UMA-Inverse"),
        name=cfg.get("run_name", "pairmixer-run"),
        mode=wandb_cfg.get("mode", "offline"),
        save_dir="logs/wandb",
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)
    torch.set_float32_matmul_precision("high")

    # ── Paths ──────────────────────────────────────────────────────────────────
    train_json = hydra.utils.to_absolute_path(cfg.paths.train_json)
    valid_json = hydra.utils.to_absolute_path(cfg.paths.valid_json)
    pdb_dir = hydra.utils.to_absolute_path(cfg.paths.pdb_dir)

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
    T_max = int(cfg.training.get("T_max", 50_000))

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
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=int(cfg.training.get("early_stop_patience", 10)),
        mode="min",
        verbose=True,
    )
    callbacks = [checkpoint_cb, lr_monitor, early_stop]

    # ── Trainer ────────────────────────────────────────────────────────────────
    accelerator = str(cfg.training.accelerator)
    precision = str(cfg.training.precision)
    if accelerator == "gpu" and not torch.cuda.is_available():
        accelerator = "cpu"
        precision = "32-true"

    max_epochs = int(cfg.training.epochs)
    if cfg.get("trainer") and cfg.trainer.get("max_epochs"):
        max_epochs = int(cfg.trainer.max_epochs)

    # Create log dirs expected by WandB / CSV logger
    os.makedirs("logs/wandb", exist_ok=True)
    os.makedirs("logs/SLURM_out", exist_ok=True)
    os.makedirs("logs/SLURM_err", exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=int(cfg.training.devices),
        precision=precision,
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        callbacks=callbacks,
        logger=_build_logger(cfg),
        default_root_dir=".",
        num_sanity_val_steps=2,
    )

    ckpt_path = None
    if cfg.get("trainer") and cfg.trainer.get("resume_from_checkpoint"):
        ckpt_path = cfg.trainer.resume_from_checkpoint

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
