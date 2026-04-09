from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.uma_inverse import UMAInverse


class UMAInverseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: Dict[str, Any],
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = UMAInverse(model_config)
        torch.set_float32_matmul_precision("high")

        if compile_model:
            try:
                self.model = torch.compile(self.model, dynamic=True)
            except Exception:
                pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def _compute_loss_and_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self(batch)
        logits = outputs["logits"]

        valid_mask = batch["residue_mask"].bool() & batch["design_mask"].bool()
        target = batch["sequence"].clone()
        target[~valid_mask] = -100

        loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100)

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = ((pred == batch["sequence"]) & valid_mask).sum()
            denom = valid_mask.sum().clamp_min(1)
            acc = correct.float() / denom.float()

        return {"loss": loss, "acc": acc}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self(batch)
        if batch_idx == 0 and self.current_epoch == 0:
            print(f"\n[Pilot Shape Check] Z tensor final shape: {out['pair_repr'].shape}\n")
        
        logits = out["logits"]
        valid_mask = batch["residue_mask"].bool() & batch["design_mask"].bool()
        target = batch["sequence"].clone()
        target[~valid_mask] = -100

        loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100)

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = ((pred == batch["sequence"]) & valid_mask).sum()
            denom = valid_mask.sum().clamp_min(1)
            acc = correct.float() / denom.float()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self._compute_loss_and_metrics(batch)
        self.log("val/loss", out["loss"], prog_bar=True, sync_dist=False)
        self.log("val/acc", out["acc"], prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=10000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
