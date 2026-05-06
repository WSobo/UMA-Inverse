"""Tests for PairMixerBlock and its triangle multiplication sub-modules."""

import pytest
import torch

from src.models.pairmixer_block import (
    PairMixerBlock,
    PairTransition,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

# ──────────────────────────────────────────────────────────────────────────────
# PairTransition
# ──────────────────────────────────────────────────────────────────────────────

def test_pair_transition_shape():
    t = PairTransition(dim=32, transition_mult=4, dropout=0.0)
    z = torch.randn(2, 10, 10, 32)
    out = t(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_pair_transition_different_mult():
    t = PairTransition(dim=64, transition_mult=2)
    z = torch.randn(1, 8, 8, 64)
    assert t(z).shape == z.shape


# ──────────────────────────────────────────────────────────────────────────────
# Triangle Multiplication
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming])
def test_triangle_mul_shape(cls):
    mod = cls(dim=32, hidden_dim=32)
    z = torch.randn(2, 12, 12, 32)
    out = mod(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming])
def test_triangle_mul_mask_zeroes_padding(cls):
    """Padding entries (mask=0) should not bleed into valid outputs."""
    torch.manual_seed(7)
    mod = cls(dim=16, hidden_dim=16)
    z = torch.randn(1, 6, 6, 16)

    # Mask out the last row/col (indices 4, 5)
    mask_partial = torch.ones(1, 6, 6)
    mask_partial[:, 4:, :] = 0
    mask_partial[:, :, 4:] = 0

    # Baseline: original z with partial mask
    out_baseline = mod(z, mask=mask_partial)

    # Perturb the masked region — valid outputs should not change
    z_perturbed = z.clone()
    z_perturbed[:, 4:, :, :] += 1e6
    z_perturbed[:, :, 4:, :] += 1e6

    out_perturbed = mod(z_perturbed, mask=mask_partial)

    # Valid region [0:4, 0:4] should match (within float tolerance)
    assert torch.allclose(out_baseline[:, :4, :4, :], out_perturbed[:, :4, :4, :], atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# PairMixerBlock
# ──────────────────────────────────────────────────────────────────────────────

def test_pairmixer_block_forward_shape():
    torch.manual_seed(0)
    block = PairMixerBlock(pair_dim=32, hidden_dim=32)
    z = torch.randn(2, 12, 12, 32)
    mask = torch.ones(2, 12, 12)
    out = block(z, mask)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_pairmixer_block_no_mask():
    """Block should run correctly when mask is None."""
    block = PairMixerBlock(pair_dim=16, hidden_dim=16, transition_mult=2)
    z = torch.randn(1, 8, 8, 16)
    out = block(z, mask=None)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_pairmixer_block_gradient_flow():
    """Loss should produce finite gradients through all parameters."""
    block = PairMixerBlock(pair_dim=16, hidden_dim=16, transition_mult=2, dropout=0.0)
    z = torch.randn(1, 6, 6, 16, requires_grad=True)
    out = block(z)
    loss = out.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


def test_pairmixer_block_residual_not_identity():
    """Output should differ from input — residual connection must add a non-zero update."""
    torch.manual_seed(42)
    block = PairMixerBlock(pair_dim=32, hidden_dim=32)
    z = torch.randn(1, 10, 10, 32)
    out = block(z)
    assert not torch.allclose(out, z, atol=1e-6), "Block output should not equal input"


@pytest.mark.parametrize("N,pair_dim,hidden_dim", [(4, 16, 16), (20, 64, 32), (30, 32, 64)])
def test_pairmixer_block_various_sizes(N, pair_dim, hidden_dim):
    block = PairMixerBlock(pair_dim=pair_dim, hidden_dim=hidden_dim)
    z = torch.randn(1, N, N, pair_dim)
    out = block(z)
    assert out.shape == (1, N, N, pair_dim)
    assert torch.isfinite(out).all()
