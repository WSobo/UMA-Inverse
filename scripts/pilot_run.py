"""1-batch overfit sanity check (pilot run).

Trains on a single batch for 100 epochs with overfit_batches=1.  Loss should
approach zero, confirming the model can memorise a single example.

Usage:
    make pilot                              # via srun on GPU
    make pilot-64                           # max_total_nodes=64
    uv run python scripts/pilot_run.py     # local (CPU fallback)
"""
import logging
import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import CSVLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.datamodule import UMAInverseDataModule
from src.training.lightning_module import UMAInverseLightningModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("UMA-Inverse Pilot Run (1-batch overfit check)")
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    train_json = hydra.utils.to_absolute_path(cfg.paths.train_json)
    valid_json = hydra.utils.to_absolute_path(cfg.paths.valid_json)
    pdb_dir    = hydra.utils.to_absolute_path(cfg.paths.pdb_dir)

    logger.info("train_json : %s", train_json)
    logger.info("pdb_dir    : %s", pdb_dir)

    datamodule = UMAInverseDataModule(
        train_json=train_json,
        valid_json=valid_json,
        pdb_dir=pdb_dir,
        batch_size=1,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        ligand_context_atoms=int(cfg.data.ligand_context_atoms),
        cutoff_for_score=float(cfg.data.cutoff_for_score),
        max_total_nodes=int(cfg.data.max_total_nodes),
    )

    model = UMAInverseLightningModule(
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        compile_model=False,
    )

    accelerator = str(cfg.training.accelerator)
    precision   = str(cfg.training.precision)
    if accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU not available — falling back to CPU")
        accelerator = "cpu"
        precision   = "32-true"

    pilot_logger = CSVLogger(
        save_dir=os.path.join(PROJECT_ROOT, "logs", "pilot"),
        name="pilot",
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        overfit_batches=1,
        logger=pilot_logger,
        callbacks=[RichProgressBar()],
        enable_checkpointing=False,
        log_every_n_steps=10,
    )

    logger.info("Beginning 1-batch convergence check ...")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Pilot run complete. Loss should be near zero.")


if __name__ == "__main__":
    main()
