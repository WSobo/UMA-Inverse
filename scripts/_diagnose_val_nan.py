"""Find valid.json PDBs that produce NaN logits/loss under stage-1 conditions.

Replays the exact stage-1 forward path (bf16-mixed, max_total_nodes=64) over
the entire valid set and prints any PDB whose forward produces non-finite
output. We use a randomly-initialized UMAInverse with the v2 config — that's
not the same model state as the trained stage-1 ep0 ckpt, but if the input
data itself is the trigger (degenerate backbone coords, weird elements),
random init will surface it too. False negatives are possible if the NaN was
weight-state-dependent.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse")

import torch
from omegaconf import OmegaConf

from src.data.datamodule import UMAInverseDataModule
from src.models.uma_inverse import UMAInverse

torch.manual_seed(0)
cfg = OmegaConf.load("/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/configs/config.yaml")

dm = UMAInverseDataModule(
    train_json="/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/LigandMPNN/training/train.json",
    valid_json="/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/LigandMPNN/training/valid.json",
    pdb_dir="/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/data/raw/pdb_archive",
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    ligand_context_atoms=25,
    cutoff_for_score=8.0,
    max_total_nodes=64,
    ligand_featurizer="atomic_number_embedding",
    residue_anchor="cb",
    return_backbone_coords=True,
)
dm.setup("fit")
ds = dm.valid_dataset
N = len(ds)
print(f"Scanning valid_dataset (N={N}) under bf16-mixed at max_total_nodes=64...")

model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
model_cfg["dropout"] = 0.0
model_cfg["gradient_checkpointing"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UMAInverse(model_cfg).to(device).eval()
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=20)

from src.data.datamodule import collate_batch
nan_ids: list[tuple[int, str, str]] = []  # (idx, pdb_id, reason)
checked = 0
for idx in range(N):
    try:
        item = ds[idx]
    except Exception as e:
        nan_ids.append((idx, "?", f"getitem_error:{type(e).__name__}"))
        continue
    if item["residue_coords"].shape[0] == 0:
        nan_ids.append((idx, item.get("pdb_id", "?"), "zero_residues"))
        continue
    pdb_id = item["pdb_id"]
    batch = collate_batch([item])
    # Move to device
    batch_d = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch_d)
    logits = out["logits"]
    seq = batch_d["sequence"]
    B, L, V = logits.shape
    loss = loss_fn(logits.view(B * L, V).float(), seq.view(B * L))
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        nan_ids.append((idx, pdb_id, f"loss={loss.item()}_logits_finite={torch.isfinite(logits).all().item()}"))
    checked += 1
    if checked % 500 == 0:
        print(f"  ... scanned {checked}/{N}, NaN hits so far: {len(nan_ids)}")

print(f"\nTotal scanned: {checked}/{N}")
print(f"NaN-triggering PDBs: {len(nan_ids)}")
for idx, pid, reason in nan_ids[:50]:
    print(f"  [idx={idx:>5}] {pid}: {reason}")
if len(nan_ids) > 50:
    print(f"  ... and {len(nan_ids) - 50} more")

# Write to disk for later use as a blacklist supplement
out_path = "/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/val_nan_pdbs.txt"
with open(out_path, "w") as f:
    for idx, pid, reason in nan_ids:
        f.write(f"{pid}\t{reason}\n")
print(f"\nWritten: {out_path}")
