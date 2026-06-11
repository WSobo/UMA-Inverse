"""Distogram auxiliary head + shared utilities.

Single source of truth for:
  * The AF3-style Cβ-Cβ bin grid (38 bins, 3.15-50.75 Å, 1.25 Å wide).
  * Virtual-Cβ derivation from N/Cα/C (matches uma_inverse.py line ~421).
  * The linear probe head (Linear(pair_dim, n_bins)).
  * The in-training auxiliary loss (compute_distogram_loss) that v5
    Phase A wires into the Lightning module.

The offline distogram probe at scripts/paper/distogram_probe.py imports
from here so the in-training metric is directly comparable to the v3
probe verdict (top-1 0.266, encoder_weak).

The head and loss are training-only: at inference time the model returns
its sequence logits only and this module is not touched. When loading a
v5 checkpoint that contains aux-head weights into an older trunk, pass
strict=False.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

# ── Bin grid ─────────────────────────────────────────────────────────────────

BIN_LO: float = 3.15
BIN_HI: float = 50.75
N_BINS: int = 38
BIN_WIDTH: float = (BIN_HI - BIN_LO) / N_BINS  # 1.25 Å


def bin_distances(dists: Tensor) -> Tensor:
    """Bin a continuous distance tensor into N_BINS classes.

    Bins 0..N_BINS-1 evenly span [BIN_LO, BIN_HI]. Distances < BIN_LO clamp
    to bin 0; distances > BIN_HI clamp to bin N_BINS-1. Returns long.
    """
    idx = ((dists - BIN_LO) / BIN_WIDTH).long()
    return idx.clamp(min=0, max=N_BINS - 1)


def bin_centers(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Center distance for each bin, used for expected-distance MAE."""
    return torch.linspace(
        BIN_LO + BIN_WIDTH / 2.0,
        BIN_HI - BIN_WIDTH / 2.0,
        N_BINS,
        device=device,
        dtype=dtype,
    )


# ── Virtual Cβ ───────────────────────────────────────────────────────────────


def derive_cb(bb: Tensor) -> Tensor:
    """Virtual Cβ from backbone N/Cα/C.

    bb: [..., 4, 3] with order (N, Cα, C, O). The fourth atom is ignored.
    Returns [..., 3]. Constants match uma_inverse.py (~line 421) and
    LigandMPNN's _construct_virtual_cb.
    """
    n  = bb[..., 0, :]
    ca = bb[..., 1, :]
    c  = bb[..., 2, :]
    b = ca - n
    c_vec = c - ca
    a = torch.linalg.cross(b, c_vec, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


# ── Linear head ──────────────────────────────────────────────────────────────


class DistogramHead(nn.Module):
    """Single linear layer pair_dim -> n_bins.

    Tiny by design: the probe measures how much geometry the encoder
    already carries, not what a deep MLP can recover from Z_ij. The same
    capacity is used during in-training aux supervision so the val/top1
    metric stays comparable to the offline probe verdict.
    """

    def __init__(self, pair_dim: int, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.proj = nn.Linear(pair_dim, n_bins)

    def forward(self, z: Tensor) -> Tensor:
        return self.proj(z)


# ── In-training auxiliary loss ──────────────────────────────────────────────


def compute_distogram_loss(
    pair_repr: Tensor,
    backbone_coords: Tensor,
    residue_mask: Tensor,
    head: DistogramHead,
    *,
    min_seq_sep: int = 2,
) -> dict[str, Tensor]:
    """Cross-entropy on Cβ-Cβ distogram bins over the residue-residue Z_ij.

    pair_repr:        [B, N, N, pair_dim]. The residue block [:, :L, :L, :]
                      is used; ligand columns are ignored.
    backbone_coords:  [B, L, 4, 3] in (N, Cα, C, O) order. Required.
    residue_mask:     [B, L] bool — True for real residues.

    Returns a dict with:
      loss     : scalar CE loss over masked residue pairs.
      top1     : scalar fraction of pairs whose argmax matches the target.
      mae      : scalar mean absolute error between expected distance and
                 the binned ground truth (in Å, computed via bin centres).
      n_pairs  : scalar long count of valid pairs used in the loss.

    Pair-validity mask: both i and j are valid residues AND |i-j| >= min_seq_sep
    (skip the diagonal + immediate neighbours, matching the offline probe).
    """
    B, L = residue_mask.shape
    device = pair_repr.device

    # Slice the residue-residue block. The encoder enforces symmetry over
    # pair tensors so an explicit symmetrisation is cheap and avoids any
    # asymmetric noise leaking into the supervision target.
    z_rr = pair_repr[:, :L, :L, :]
    z_rr = 0.5 * (z_rr + z_rr.transpose(1, 2))

    # Target distances from virtual Cβ; backbone_coords arrives in the
    # model's working dtype (often bf16) — Cβ derivation must be in fp32
    # so the cross product is stable.
    cb = derive_cb(backbone_coords.float())                  # [B, L, 3]
    dists = torch.cdist(cb, cb)                              # [B, L, L]
    target = bin_distances(dists)                            # [B, L, L] long

    # Validity: both residues exist AND |i-j| >= min_seq_sep.
    rmask = residue_mask.bool()
    pair_valid = rmask[:, :, None] & rmask[:, None, :]       # [B, L, L]
    arange = torch.arange(L, device=device)
    seq_sep_ok = (arange[:, None] - arange[None, :]).abs() >= int(min_seq_sep)
    pair_valid = pair_valid & seq_sep_ok[None, :, :]

    logits = head(z_rr)                                       # [B, L, L, n_bins]

    flat_logits = logits[pair_valid]                          # [P, n_bins]
    flat_target = target[pair_valid]                          # [P]
    n_pairs = flat_target.numel()

    if n_pairs == 0:
        zero = torch.zeros((), device=device, dtype=pair_repr.dtype)
        return {
            "loss": zero,
            "top1": zero,
            "mae": zero,
            "n_pairs": torch.zeros((), device=device, dtype=torch.long),
        }

    # CE in fp32 for numerical stability when pair_repr is bf16.
    loss = torch.nn.functional.cross_entropy(flat_logits.float(), flat_target)

    with torch.no_grad():
        probs = torch.softmax(flat_logits.float(), dim=-1)
        pred = probs.argmax(dim=-1)
        top1 = (pred == flat_target).float().mean()
        centers = bin_centers(device, torch.float32)
        exp_dist = (probs * centers).sum(dim=-1)
        true_dist = centers[flat_target]
        mae = (exp_dist - true_dist).abs().mean()

    return {
        "loss": loss,
        "top1": top1,
        "mae": mae,
        "n_pairs": torch.tensor(n_pairs, device=device, dtype=torch.long),
    }
