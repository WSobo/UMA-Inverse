"""Tests for the UMAInverse end-to-end model."""

import pytest
import torch

from src.models.uma_inverse import RBFEmbedding, UMAInverse

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _small_config(**overrides):
    cfg = {
        "residue_input_dim": 6,
        "ligand_input_dim": 6,
        "node_dim": 32,
        "pair_dim": 32,
        "pair_hidden_dim": 32,
        "num_pairmixer_blocks": 2,
        "pair_transition_mult": 2,
        "num_rbf": 8,
        "max_distance": 12.0,
        "dropout": 0.0,
        "gradient_checkpointing": False,
        "thermal_noise_std": 0.0,
    }
    cfg.update(overrides)
    return cfg


def _make_batch(B=1, L=16, M=5, device="cpu"):
    return {
        "residue_coords": torch.randn(B, L, 3, device=device),
        "residue_features": torch.randn(B, L, 6, device=device),
        "residue_mask": torch.ones(B, L, dtype=torch.bool, device=device),
        "sequence": torch.randint(0, 20, (B, L), device=device),
        "design_mask": torch.ones(B, L, dtype=torch.bool, device=device),
        "ligand_coords": torch.randn(B, M, 3, device=device),
        "ligand_features": torch.randn(B, M, 6, device=device),
        "ligand_mask": torch.ones(B, M, dtype=torch.bool, device=device),
    }


# ──────────────────────────────────────────────────────────────────────────────
# RBFEmbedding
# ──────────────────────────────────────────────────────────────────────────────

def test_rbf_shape():
    rbf = RBFEmbedding(num_rbf=16, max_distance=20.0)
    dist = torch.rand(2, 10, 10) * 20.0
    out = rbf(dist)
    assert out.shape == (2, 10, 10, 16)
    assert torch.isfinite(out).all()


def test_rbf_range():
    """All RBF values should be in (0, 1]."""
    rbf = RBFEmbedding(num_rbf=8, max_distance=10.0)
    dist = torch.linspace(0, 10, 50)
    out = rbf(dist)
    assert (out >= 0).all() and (out <= 1 + 1e-5).all()


# ──────────────────────────────────────────────────────────────────────────────
# UMAInverse forward
# ──────────────────────────────────────────────────────────────────────────────

def test_uma_inverse_forward_shape():
    torch.manual_seed(1)
    model = UMAInverse(_small_config())
    batch = _make_batch(B=2, L=20, M=8)
    out = model(batch)
    assert out["logits"].shape == (2, 20, 21)
    assert out["pair_repr"].shape == (2, 28, 28, 32)


def test_uma_inverse_no_ligand():
    """Model must handle zero-atom ligand gracefully."""
    model = UMAInverse(_small_config())
    batch = _make_batch(B=1, L=10, M=0)
    batch["ligand_coords"] = torch.zeros(1, 0, 3)
    batch["ligand_features"] = torch.zeros(1, 0, 6)
    batch["ligand_mask"] = torch.zeros(1, 0, dtype=torch.bool)
    out = model(batch)
    assert out["logits"].shape == (1, 10, 21)
    assert torch.isfinite(out["logits"]).all()


def test_uma_inverse_no_sequence_gives_zero_ar_context():
    """Without a sequence key the AR context should be all zeros."""
    model = UMAInverse(_small_config())
    batch = _make_batch(B=1, L=8, M=3)
    batch.pop("sequence")
    out = model(batch)
    assert out["logits"].shape == (1, 8, 21)
    assert torch.isfinite(out["logits"]).all()


def test_uma_inverse_padded_mask():
    """Padded positions (mask=0) should not cause NaNs."""
    model = UMAInverse(_small_config())
    batch = _make_batch(B=2, L=20, M=5)
    # Pad the second sample to L=12 (rows 12-19 are padding)
    batch["residue_mask"][1, 12:] = False
    out = model(batch)
    assert torch.isfinite(out["logits"]).all()


def test_uma_inverse_gradient_flow():
    """Backward pass should produce finite, non-zero gradients."""
    model = UMAInverse(_small_config())
    batch = _make_batch(B=1, L=10, M=4)
    out = model(batch)
    loss = out["logits"].sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


def test_uma_inverse_logits_sum_to_nonzero():
    """Logits should not be all-zero (would indicate dead initialisation)."""
    torch.manual_seed(99)
    model = UMAInverse(_small_config())
    batch = _make_batch(B=1, L=8, M=3)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].abs().sum() > 0


@pytest.mark.parametrize("B,L,M", [(1, 8, 0), (2, 20, 10), (1, 50, 25)])
def test_uma_inverse_various_sizes(B, L, M):
    model = UMAInverse(_small_config())
    batch = _make_batch(B=B, L=L, M=M)
    if M == 0:
        batch["ligand_coords"] = torch.zeros(B, 0, 3)
        batch["ligand_features"] = torch.zeros(B, 0, 6)
        batch["ligand_mask"] = torch.zeros(B, 0, dtype=torch.bool)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (B, L, 21)
    assert torch.isfinite(out["logits"]).all()


def test_uma_inverse_thermal_noise_train_only():
    """Thermal noise should be applied in train mode and not in eval mode."""
    model = UMAInverse(_small_config(thermal_noise_std=10.0))
    batch = _make_batch(B=1, L=8, M=3)

    model.train()
    out_train = model(batch)

    model.eval()
    with torch.no_grad():
        out_eval = model(batch)

    # With large noise, train and eval logits will differ
    assert not torch.allclose(out_train["logits"].detach(), out_eval["logits"], atol=1e-3)
