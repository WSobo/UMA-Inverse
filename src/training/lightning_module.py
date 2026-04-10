"""UMA-Inverse Lightning training module."""
import logging
import math
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.uma_inverse import UMAInverse

logger = logging.getLogger(__name__)


def _warmup_cosine_lambda(warmup_steps: int, T_max: int):
    """Linear warmup then cosine decay, both expressed as a LambdaLR multiplier."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(T_max - warmup_steps)
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

        if warmup_steps >= T_max:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be strictly less than "
                f"T_max ({T_max}). The cosine schedule would be undefined otherwise."
            )

        self.save_hyperparameters()

        self.model = UMAInverse(model_config)
        torch.set_float32_matmul_precision("high")

        if compile_model:
            try:
                self.model = torch.compile(self.model, dynamic=True)
                logger.info("torch.compile enabled (dynamic=True)")
            except Exception as e:
                logger.warning(
                    "torch.compile failed — continuing with uncompiled model. Reason: %s", e
                )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    # ── Shared loss computation ───────────────────────────────────────────────

    def _compute_loss_and_metrics(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run one forward pass and return loss + accuracy over designed residues."""
        outputs = self(batch)
        logits = outputs["logits"]

        valid_mask = batch["residue_mask"].bool() & batch["design_mask"].bool()
        target = batch["sequence"].clone()
        target[~valid_mask] = -100  # cross_entropy ignores index -100

        loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100)

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = ((pred == batch["sequence"]) & valid_mask).sum()
            denom = valid_mask.sum().clamp_min(1)
            acc = correct.float() / denom.float()

        return {"loss": loss, "acc": acc}

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Deterministic-per-(epoch, batch) decoding order: non-designed residues
        # decode first so designed ones condition on fixed context.
        # Seeded by (current_epoch * 1e6 + batch_idx) for reproducibility.
        if "decoding_order" not in batch:
            B, L = batch["residue_mask"].shape
            seed = self.current_epoch * 1_000_000 + batch_idx
            g = torch.Generator(device=batch["residue_mask"].device)
            g.manual_seed(seed)
            orders = []
            for i in range(B):
                rand_val = torch.rand(L, generator=g, device=batch["residue_mask"].device)
                rand_val[~batch["design_mask"][i].bool()] -= 100.0
                orders.append(torch.argsort(torch.argsort(rand_val)))
            batch["decoding_order"] = torch.stack(orders)

        metrics = self._compute_loss_and_metrics(batch)
        loss = metrics["loss"]
        acc  = metrics["acc"]

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  acc,  on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            prog_bar=False,
        )
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        total_norm = sum(
            p.grad.detach().float().norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        ) ** 0.5
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False)

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self._compute_loss_and_metrics(batch)
        self.log("val/loss", out["loss"], prog_bar=True, sync_dist=False)
        self.log("val/acc",  out["acc"],  prog_bar=True, sync_dist=False)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────

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
