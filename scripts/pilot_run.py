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
        # v2 flags must match the model's forward expectations — defaults
        # to v1 here would yield a KeyError when the config flips on v2 since
        # the DataModule would emit ligand_features while the model asks for
        # ligand_atomic_numbers.
        ligand_featurizer=str(cfg.data.get("ligand_featurizer", "onehot6")),
        residue_anchor=str(cfg.data.get("residue_anchor", "ca")),
        return_backbone_coords=(
            str(cfg.data.get("pair_distance_atoms", "anchor_only")) == "backbone_full"
            or str(cfg.data.get("pair_distance_atoms_ligand", "anchor_only")) == "backbone_full"
            or bool(cfg.data.get("frame_relative_angles", False))
        ),
        # v3 flags — defaults off, mirror scripts/train.py.
        return_frame_relative_angles=bool(cfg.data.get("frame_relative_angles", False)),
        sidechain_context_rate=float(cfg.data.get("sidechain_context_rate", 0.0)),
        aug_seed=int(cfg.get("seed", 0)),
    )

    # Pilot-specific overrides:
    # 1. LR schedule — default (warmup=500, T_max=50k) is tuned for full training
    #    and never escapes warmup in a 100-step pilot (peak LR ≈ 6e-5 ≈ 5% of
    #    nominal). Scale to the pilot budget.
    # 2. Regularization off — dropout and weight_decay prevent a 1-batch overfit
    #    from converging to near-zero loss (the whole point of the sanity check).
    #    Gradient checkpointing is an AMP/autograd interaction risk on tiny
    #    models; disable for the pilot to remove an unknown from the result.
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["dropout"] = 0.0
    model_cfg["gradient_checkpointing"] = False

    # Pilot budget — override via `++pilot.epochs=500` on the CLI.
    # LR schedule (warmup + T_max) scales with the epoch budget.
    pilot_epochs = int(OmegaConf.select(cfg, "pilot.epochs", default=100))
    pilot_warmup = max(10, pilot_epochs // 10)

    model = UMAInverseLightningModule(
        model_config=model_cfg,
        lr=float(cfg.training.lr),
        weight_decay=0.0,
        warmup_steps=pilot_warmup,
        T_max=pilot_epochs,
        compile_model=False,
        fixed_decoding_order=True,
    )

    accelerator = str(cfg.training.accelerator)
    # Pilot always runs in fp32: bf16-mixed loses too much precision for the
    # small gradients needed to drive a 1-batch overfit to near-zero loss.
    precision = "32-true"
    if accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU not available — falling back to CPU")
        accelerator = "cpu"

    pilot_logger = CSVLogger(
        save_dir=os.path.join(PROJECT_ROOT, "logs", "pilot"),
        name="pilot",
    )

    # Read devices from config so DDP (stage 2/3) semantics can be rehearsed
    # in the pilot by setting ++training.devices=2 at submit time. devices=1
    # is the historical pilot default and stays single-GPU.
    pilot_devices = int(cfg.training.get("devices", 1))
    trainer = pl.Trainer(
        max_epochs=pilot_epochs,
        accelerator=accelerator,
        devices=pilot_devices,
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
