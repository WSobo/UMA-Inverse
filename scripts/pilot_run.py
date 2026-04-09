import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.datamodule import UMAInverseDataModule
from src.training.lightning_module import UMAInverseLightningModule

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("🚀 Initiating UMA-Inverse Pilot Run (Sanity & Overfit Check)...")
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    train_json = hydra.utils.to_absolute_path(cfg.paths.train_json)
    valid_json = hydra.utils.to_absolute_path(cfg.paths.valid_json)
    pdb_dir = hydra.utils.to_absolute_path(cfg.paths.pdb_dir)

    print("Loading DataModule...")
    datamodule = UMAInverseDataModule(
        train_json=train_json,
        valid_json=valid_json,
        pdb_dir=pdb_dir,
        batch_size=1,  # Force batch size 1 for single-batch overfitting
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        ligand_context_atoms=int(cfg.data.ligand_context_atoms),
        cutoff_for_score=float(cfg.data.cutoff_for_score),
        max_total_nodes=int(cfg.data.max_total_nodes),
    )

    print("Loading UMA-Inverse Model...")
    model = UMAInverseLightningModule(
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        compile_model=False,
    )

    print("Setting up Trainer in overfit_batches mode...")
    accelerator = str(cfg.training.accelerator)
    precision = str(cfg.training.precision)
    if accelerator == "gpu" and not torch.cuda.is_available():
        accelerator = "cpu"
        precision = "32-true"

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        overfit_batches=1, # <--- The Magic Flag
        logger=False,      # Don't pollute your Weights & Biases dashboard
        enable_checkpointing=False
    )

    print("Beginning 1-Batch Convergence Check...")
    trainer.fit(model, datamodule=datamodule)
    print("✅ Pilot run complete! Sequence loss should be near zero.")

if __name__ == "__main__":
    main()
