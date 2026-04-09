import torch

from src.models.pairmixer_block import PairMixerBlock


def test_pairmixer_block_forward_shape():
    torch.manual_seed(0)
    block = PairMixerBlock(pair_dim=32, hidden_dim=32)
    z = torch.randn(2, 12, 12, 32)
    mask = torch.ones(2, 12, 12)

    out = block(z, mask)

    assert out.shape == z.shape
    assert torch.isfinite(out).all()
