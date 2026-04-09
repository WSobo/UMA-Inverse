import math
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.uma_inverse import UMAInverse


def _warmup_cosine_lambda(warmup_steps: int, T_max: int):
    """Linear warmup then cosine decay, both expressed as a LambdaLR multiplier."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, T_max - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))
    return lr_lambda


class UMAInverseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: Dict[str, Any],
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 500,
        T_max: int = 50_000,
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

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    # ──────────────────────────────────────────────────────────────────────────
    # Shared loss computation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_loss_and_metrics(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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

    # ──────────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Randomised decoding order: non-designed residues decode first so
        # designed residues can condition on fixed context.
        if "decoding_order" not in batch:
            B, L = batch["residue_mask"].shape
            orders = []
            for i in range(B):
                rand_val = torch.rand(L, device=batch["residue_mask"].device)
                rand_val[~batch["design_mask"][i].bool()] -= 100.0
                orders.append(torch.argsort(torch.argsort(rand_val)))
            batch["decoding_order"] = torch.stack(orders)

        out = self(batch)

        if batch_idx == 0 and self.current_epoch == 0:
            print(
                f"\n[Shape Check] pair_repr: {tuple(out['pair_repr'].shape)}, "
                f"logits: {tuple(out['logits'].shape)}\n"
            )

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
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=False)

        # Log gradient norm after backward (Lightning exposes this via the hook)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        # Log total gradient norm for monitoring training stability
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().float().norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Validation step
    # ──────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self._compute_loss_and_metrics(batch)
        self.log("val/loss", out["loss"], prog_bar=True, sync_dist=False)
        self.log("val/acc", out["acc"], prog_bar=True, sync_dist=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Optimiser + scheduler
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=_warmup_cosine_lambda(
                warmup_steps=self.hparams.warmup_steps,
                T_max=self.hparams.T_max,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
