"""CPU-only diagnostic: is the AR teacher-forcing path actually contributing signal?

Loads one cached sample, runs the model on CPU with (a) teacher forcing ON and
(b) teacher forcing OFF (sequence=None). Compares loss on a single sample across
100 pseudo-training iterations using Adam. If loss plateaus at the SAME value in
both runs, the AR path is not contributing → architectural fix required.

Usage:
    uv run python scripts/diag_ar_path.py
"""
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data.datamodule import collate_batch
from src.models.uma_inverse import UMAInverse


def _load_one(pdb_id: str = "106m") -> dict:
    path = os.path.join(PROJECT_ROOT, "data", "processed", f"{pdb_id}.pt")
    item = torch.load(path, map_location="cpu", weights_only=True)
    item["pdb_id"] = pdb_id
    return item


def _make_decoding_order(residue_mask: torch.Tensor, seed: int) -> torch.Tensor:
    B, L = residue_mask.shape
    g = torch.Generator()
    g.manual_seed(seed)
    rand_val = torch.rand(L, generator=g)
    return torch.argsort(torch.argsort(rand_val)).unsqueeze(0).expand(B, L).contiguous()


def _overfit(model: UMAInverse, batch: dict, *, teacher_force: bool, n_iters: int = 100) -> list:
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    model.train()
    losses = []
    for it in range(n_iters):
        b = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        b["decoding_order"] = _make_decoding_order(b["residue_mask"], seed=it)
        if not teacher_force:
            b["sequence"] = None
        out = model(b)
        logits = out["logits"]
        valid = batch["residue_mask"].bool() & batch["design_mask"].bool()
        target = batch["sequence"].clone()
        target[~valid] = -100
        loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    return losses


def main() -> None:
    torch.manual_seed(42)

    # Use small model (matches tests/conftest.small_model_config) for CPU speed
    cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["num_pairmixer_blocks"] = 2     # shallower for CPU
    model_cfg["node_dim"] = 64
    model_cfg["pair_dim"] = 64
    model_cfg["pair_hidden_dim"] = 64
    model_cfg["gradient_checkpointing"] = False

    item = _load_one("106m")
    batch = collate_batch([item])

    print(f"Sample: pdb=106m L={item['residue_coords'].shape[0]} "
          f"M={item['ligand_coords'].shape[0]}")
    print(f"design_mask.sum()/residue_mask.sum() = "
          f"{int(batch['design_mask'].sum())}/{int(batch['residue_mask'].sum())}")
    print(f"design_mask all True: {bool(batch['design_mask'].all())}")
    print()

    # Re-seed for fair comparison between the two models
    torch.manual_seed(42)
    model_tf = UMAInverse(model_cfg)
    torch.manual_seed(42)
    model_no = UMAInverse(model_cfg)

    print("Run A: teacher forcing ON (normal training)")
    losses_tf = _overfit(model_tf, batch, teacher_force=True, n_iters=100)

    print("Run B: teacher forcing OFF (sequence=None)")
    losses_no = _overfit(model_no, batch, teacher_force=False, n_iters=100)

    print()
    print(f"{'iter':>6} {'TF-on loss':>12} {'TF-off loss':>12}")
    for it in [0, 9, 19, 49, 99]:
        print(f"{it:>6d} {losses_tf[it]:>12.4f} {losses_no[it]:>12.4f}")
    print()
    print(f"Final TF-on:  {losses_tf[-1]:.4f}")
    print(f"Final TF-off: {losses_no[-1]:.4f}")
    gap = losses_no[-1] - losses_tf[-1]
    print(f"Gap (TF-off - TF-on): {gap:.4f}")
    if gap < 0.2:
        print()
        print("→ AR path contributes <0.2 nats of loss. Teacher forcing is NOT")
        print("  providing useful signal. The 1-scalar AR gate is the bottleneck.")
    else:
        print()
        print(f"→ AR path contributes {gap:.2f} nats — teacher forcing is working.")
        print("  Plateau cause is elsewhere (node readout? encoder capacity?).")


if __name__ == "__main__":
    main()
