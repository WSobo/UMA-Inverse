"""Unit tests for the v5 Phase A distogram aux head and helpers."""
import pytest
import torch

from src.training.distogram import (
    BIN_HI,
    BIN_LO,
    BIN_WIDTH,
    N_BINS,
    DistogramHead,
    bin_centers,
    bin_distances,
    compute_distogram_loss,
    derive_cb,
)

# ── Bin grid ────────────────────────────────────────────────────────────────


def test_bin_grid_constants():
    """Bin grid must match the AF3 / v3 probe schedule exactly.

    BIN_WIDTH is the derived value (BIN_HI - BIN_LO) / N_BINS ≈ 1.2526 Å,
    not the rounded 1.25 quoted in the v3 probe docstring. Carrying the
    rounded version here would mean the in-training aux head bins
    distances differently from the offline probe.
    """
    assert N_BINS == 38
    assert BIN_LO == pytest.approx(3.15)
    assert BIN_HI == pytest.approx(50.75)
    assert BIN_WIDTH == pytest.approx((BIN_HI - BIN_LO) / N_BINS)
    assert BIN_WIDTH == pytest.approx(1.2526, abs=1e-4)


def test_bin_distances_boundaries():
    """Distances at and beyond the boundaries must clamp into the end bins."""
    dists = torch.tensor([0.0, BIN_LO, BIN_LO + BIN_WIDTH / 2.0, BIN_HI, BIN_HI + 10.0])
    binned = bin_distances(dists)
    assert binned.dtype == torch.long
    assert binned[0].item() == 0
    assert binned[1].item() == 0
    assert binned[2].item() == 0
    assert binned[3].item() == N_BINS - 1
    assert binned[4].item() == N_BINS - 1


def test_bin_distances_monotonic():
    """Sorted distances → sorted bin indices."""
    dists = torch.linspace(0.0, BIN_HI + 5.0, 200)
    binned = bin_distances(dists)
    diff = binned[1:] - binned[:-1]
    assert (diff >= 0).all()


def test_bin_centers_shape_and_endpoints():
    centers = bin_centers(torch.device("cpu"), torch.float32)
    assert centers.shape == (N_BINS,)
    assert centers[0].item() == pytest.approx(BIN_LO + BIN_WIDTH / 2.0)
    assert centers[-1].item() == pytest.approx(BIN_HI - BIN_WIDTH / 2.0)


# ── Virtual Cβ ──────────────────────────────────────────────────────────────


def test_derive_cb_returns_finite():
    """A typical residue with non-collinear N/Cα/C produces a finite Cβ."""
    # Use a textbook idealised backbone for one residue.
    bb = torch.tensor([[
        [-1.458,  0.000,  0.000],   # N
        [ 0.000,  0.000,  0.000],   # Cα
        [ 1.525,  0.000,  0.000],   # C
        [ 2.000,  1.000,  0.000],   # O (unused)
    ]])  # [1, 4, 3]
    cb = derive_cb(bb)
    assert cb.shape == (1, 3)
    assert torch.isfinite(cb).all()


def test_derive_cb_translation_equivariant():
    """Translating the backbone translates Cβ by the same vector."""
    bb = torch.randn(3, 4, 3)
    delta = torch.tensor([2.0, -1.5, 3.7])
    cb_a = derive_cb(bb)
    cb_b = derive_cb(bb + delta)
    assert torch.allclose(cb_b - cb_a, delta.expand_as(cb_a), atol=1e-5)


# ── DistogramHead ───────────────────────────────────────────────────────────


def test_distogram_head_shape():
    head = DistogramHead(pair_dim=16, n_bins=N_BINS)
    z = torch.randn(2, 8, 8, 16)
    out = head(z)
    assert out.shape == (2, 8, 8, N_BINS)


def test_distogram_head_param_count():
    """The head is deliberately tiny — Linear(pair_dim, N_BINS) with bias."""
    head = DistogramHead(pair_dim=16, n_bins=N_BINS)
    n = sum(p.numel() for p in head.parameters())
    assert n == 16 * N_BINS + N_BINS


# ── compute_distogram_loss ──────────────────────────────────────────────────


def _synth_inputs(B: int = 2, L: int = 12, pair_dim: int = 16):
    pair_repr = torch.randn(B, L + 3, L + 3, pair_dim)  # extra ligand slots ignored
    backbone = torch.randn(B, L, 4, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    return pair_repr, backbone, mask


def test_distogram_loss_runs_and_finite():
    pair_repr, backbone, mask = _synth_inputs()
    head = DistogramHead(pair_dim=pair_repr.shape[-1], n_bins=N_BINS)
    out = compute_distogram_loss(pair_repr, backbone, mask, head)
    assert {"loss", "top1", "mae", "n_pairs"} <= set(out.keys())
    assert torch.isfinite(out["loss"]).all()
    assert torch.isfinite(out["top1"]).all()
    assert torch.isfinite(out["mae"]).all()
    assert int(out["n_pairs"].item()) > 0


def test_distogram_loss_zero_pairs_when_short():
    """L=1 leaves no |i-j| >= 2 pairs — loss should be a zero scalar, not NaN."""
    pair_repr = torch.randn(1, 4, 4, 8)
    backbone = torch.randn(1, 1, 4, 3)
    mask = torch.ones(1, 1, dtype=torch.bool)
    head = DistogramHead(pair_dim=8, n_bins=N_BINS)
    out = compute_distogram_loss(pair_repr, backbone, mask, head)
    assert int(out["n_pairs"].item()) == 0
    assert out["loss"].item() == 0.0


def test_distogram_loss_respects_mask():
    """Masked-out residues must not contribute pairs to the count."""
    pair_repr, backbone, mask = _synth_inputs()
    # Mask the last 3 residues
    mask[:, -3:] = False
    head = DistogramHead(pair_dim=pair_repr.shape[-1], n_bins=N_BINS)
    out_masked = compute_distogram_loss(pair_repr, backbone, mask, head)
    out_full   = compute_distogram_loss(pair_repr, backbone, torch.ones_like(mask), head)
    assert int(out_masked["n_pairs"].item()) < int(out_full["n_pairs"].item())


def test_distogram_loss_gradient_flows_to_head():
    """Backprop into a parametrised head must produce a non-null gradient."""
    pair_repr, backbone, mask = _synth_inputs()
    head = DistogramHead(pair_dim=pair_repr.shape[-1], n_bins=N_BINS)
    out = compute_distogram_loss(pair_repr, backbone, mask, head)
    out["loss"].backward()
    assert head.proj.weight.grad is not None
    assert torch.isfinite(head.proj.weight.grad).all()
    assert head.proj.weight.grad.abs().sum().item() > 0.0


def test_distogram_loss_overfits_tiny_problem():
    """Sanity: a few hundred optim steps on one synthetic batch should drive
    the aux loss meaningfully down and top1 above the chance baseline (1/38)."""
    torch.manual_seed(0)
    pair_repr, backbone, mask = _synth_inputs(B=1, L=10, pair_dim=32)
    pair_repr = pair_repr.requires_grad_(False)
    head = DistogramHead(pair_dim=32, n_bins=N_BINS)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    initial = compute_distogram_loss(pair_repr, backbone, mask, head)["loss"].item()
    for _ in range(200):
        out = compute_distogram_loss(pair_repr, backbone, mask, head)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
    final = out["loss"].item()
    top1 = out["top1"].item()
    assert final < initial - 0.1, f"loss did not move enough: {initial:.3f} -> {final:.3f}"
    assert top1 > 1.0 / N_BINS, f"top1={top1:.3f} below chance baseline"
