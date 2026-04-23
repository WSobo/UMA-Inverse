"""Inference smoke test for UMA-Inverse.

This file is structured with ``# %%`` cell markers so it runs as a plain
Python script on SLURM *and* opens as an interactive notebook in VS Code
(no jupyter dependency required). Execute with:

    sbatch scripts/SLURM/05a_smoke_test.sh

Purpose: confirm the stage-3 ep11 checkpoint can be loaded, run the
encoder once, and produce sensible sequence designs on a real PDB.
Compares teacher-forced scoring and autoregressive sampling on a single
ligand-bound structure from the test fixtures.
"""
# ruff: noqa: E402
# %% [markdown]
# # UMA-Inverse inference smoke test
#
# Loads the best stage-3 checkpoint (epoch 11) and runs both teacher-forced
# scoring and autoregressive sampling on a fixture PDB. Reports:
#
# - overall sequence recovery (full protein)
# - recovery restricted to residues within 5 Å of a ligand heavy atom
# - autoregressive sample confidence (overall + ligand-neighbourhood)
#
# If any cell raises, the checkpoint or inference plumbing is broken.

# %%
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.inference.constraints import DesignConstraints
from src.inference.decoding import autoregressive_design, score_sequence
from src.inference.output import ligand_neighbour_mask_from_ctx
from src.inference.session import InferenceSession
from src.utils.io import ids_to_sequence

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("smoke_test")

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT  = PROJECT_ROOT / "checkpoints" / "epoch_snapshots" / "epoch-epoch=11.ckpt"
PDB_PATH    = PROJECT_ROOT / "tests" / "fixtures" / "1bc8.pdb"
LIGAND_CUTOFF_A = 5.0

print(f"project root : {PROJECT_ROOT}")
print(f"config       : {CONFIG_PATH}")
print(f"checkpoint   : {CHECKPOINT}")
print(f"test pdb     : {PDB_PATH}")
print(f"cuda avail.  : {torch.cuda.is_available()}")

# %% [markdown]
# ## Load session (encoder weights + config)

# %%
session = InferenceSession.from_checkpoint(
    config_path=CONFIG_PATH,
    checkpoint=CHECKPOINT,
    device="auto",
)
print(f"device       : {session.device}")
print(f"model params : {sum(p.numel() for p in session.model.parameters()):,}")

# %% [markdown]
# ## Encode the structure once
#
# `load_structure` runs the encoder and returns a `StructureContext` that
# downstream sampling reuses — no re-encoding per sample.

# %%
ctx = session.load_structure(
    PDB_PATH,
    ligand_cutoff=LIGAND_CUTOFF_A,
)
print(f"residues (L)       : {ctx.residue_count}")
print(f"pair tensor N      : {ctx.z.shape[1]}  (residues + ligand atoms)")
print(f"ligand atoms       : {ctx.z.shape[1] - ctx.residue_count}")
print(f"native sequence    : {ids_to_sequence(ctx.native_sequence.cpu().tolist())}")

# %% [markdown]
# ## Teacher-forced scoring
#
# Compute per-position log-probs with native sequence as context, averaged
# over 4 random decoding orders. Recovery = argmax of per-position logits
# against native — the standard inverse-folding benchmarking protocol.

# %%
score = score_sequence(
    session=session,
    ctx=ctx,
    sequence=ctx.native_sequence,
    mode="autoregressive",
    num_batches=4,
    seed=0,
)
log_probs = score.log_probs
print(f"mean log_p (nats)  : {log_probs.mean().item():+.4f}")
print(f"perplexity         : {torch.exp(-log_probs.mean()).item():.3f}")

# %% [markdown]
# ### Recovery, overall and within 5 Å of ligand
#
# For teacher-forced recovery we re-decode with `autoregressive_design` at
# near-zero temperature (argmax-equivalent) and compare against native.
# Scoring alone doesn't return argmax tokens.

# %%
constraints = DesignConstraints().resolve(ctx)
greedy = autoregressive_design(
    session=session,
    ctx=ctx,
    constraints=constraints,
    num_samples=1,
    temperature=1e-6,
    seed=0,
    decoding_order="random",
)[0]

pred_tokens = greedy.token_ids
native_tokens = ctx.native_sequence.cpu()
valid = native_tokens != 20  # exclude X
correct = (pred_tokens == native_tokens) & valid
overall_recovery = correct.sum().item() / max(1, valid.sum().item())

ligand_mask = ligand_neighbour_mask_from_ctx(ctx, cutoff=LIGAND_CUTOFF_A).cpu()
near_valid = valid & ligand_mask
near_correct = correct & ligand_mask
near_recovery = (
    near_correct.sum().item() / near_valid.sum().item()
    if near_valid.any()
    else float("nan")
)

print(f"overall recovery   : {overall_recovery:.3f}  ({correct.sum().item()}/{valid.sum().item()})")
print(f"near-ligand (≤{LIGAND_CUTOFF_A:.0f}Å): {near_recovery:.3f}  "
      f"({near_correct.sum().item()}/{near_valid.sum().item()})")

# %% [markdown]
# ## Autoregressive sampling
#
# Draw 5 sequences at T=0.1 from the same `ctx`. Each sample carries its
# own per-position probability distribution; we compare recovery and the
# model's self-reported confidence.

# %%
samples = autoregressive_design(
    session=session,
    ctx=ctx,
    constraints=constraints,
    num_samples=5,
    batch_size=5,
    temperature=0.1,
    seed=42,
    decoding_order="random",
)

print(f"{'seed':>6}  {'recovery':>9}  {'near-5Å':>9}  {'conf':>6}  {'lig-conf':>8}")
designable = ctx.design_mask.cpu()
for s in samples:
    s_correct = (s.token_ids == native_tokens) & valid
    rec = s_correct.sum().item() / max(1, valid.sum().item())
    near_rec = (
        (s_correct & ligand_mask).sum().item() / near_valid.sum().item()
        if near_valid.any()
        else float("nan")
    )
    overall_conf = s.overall_confidence(designable)
    lig_conf = s.ligand_confidence(designable, ligand_mask)
    print(f"{s.seed:>6}  {rec:>9.3f}  {near_rec:>9.3f}  {overall_conf:>6.3f}  {lig_conf:>8.3f}")

# %% [markdown]
# ## Summary
#
# A working model should show:
#
# - overall recovery in the 0.50-0.65 band (close to the stage-3 val_acc)
# - near-ligand recovery ≥ overall recovery (the model exploits ligand context)
# - sample-to-sample variance low at T=0.1 (most samples near the greedy)
# - `lig-conf` column bounded in (0, 1] — if 0.0, ligand mask is empty
#   (this fixture has a ligand, so values should be strictly positive)

# %%
top = samples[0]
print(f"native : {ids_to_sequence(native_tokens.tolist())}")
print(f"design : {ids_to_sequence(top.token_ids.tolist())}")
print(f"diff   : " + "".join(
    " " if a == b else "^"
    for a, b in zip(native_tokens.tolist(), top.token_ids.tolist())
))

print("\nsmoke test passed.")
