import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.datamodule import UMAInverseDataModule
from src.training.lightning_module import UMAInverseLightningModule


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)
    torch.set_float32_matmul_precision("high")

    train_json = hydra.utils.to_absolute_path(cfg.paths.train_json)
    valid_json = hydra.utils.to_absolute_path(cfg.paths.valid_json)
    pdb_dir = hydra.utils.to_absolute_path(cfg.paths.pdb_dir)

    datamodule = UMAInverseDataModule(
        train_json=train_json,
        valid_json=valid_json,
        pdb_dir=pdb_dir,
        batch_size=int(cfg.data.batch_size),
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
        compile_model=bool(cfg.training.compile_model),
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="uma-inverse-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    callbacks = [checkpoint_cb, LearningRateMonitor(logging_interval="step")]

    accelerator = str(cfg.training.accelerator)
    precision = str(cfg.training.precision)
    if accelerator == "gpu" and not torch.cuda.is_available():
        accelerator = "cpu"
        precision = "32-true"

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        accelerator=accelerator,
        devices=int(cfg.training.devices),
        precision=precision,
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        callbacks=callbacks,
        default_root_dir=".",
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
