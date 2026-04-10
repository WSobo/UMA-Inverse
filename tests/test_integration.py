"""End-to-end integration tests: DataModule → UMAInverse → loss.

These tests exercise the full pipeline path without touching disk or GPUs,
using synthetic in-memory data.  They catch regressions that unit tests miss —
e.g. shape mismatches between collate_batch and the model's forward pass.
"""
import torch
import torch.nn.functional as F
import pytest

from src.data.datamodule import collate_batch
from src.models.uma_inverse import UMAInverse
from src.training.lightning_module import UMAInverseLightningModule
from tests.conftest import make_sample, small_model_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _loss_from_batch(model: UMAInverse, batch: dict) -> torch.Tensor:
    """Compute cross-entropy loss over designed residues."""
    out        = model(batch)
    valid_mask = batch["residue_mask"].bool() & batch["design_mask"].bool()
    target     = batch["sequence"].clone()
    target[~valid_mask] = -100
    return F.cross_entropy(out["logits"].transpose(1, 2), target, ignore_index=-100)


# ── Core pipeline tests ───────────────────────────────────────────────────────

class TestPipelineEndToEnd:
    def test_collate_then_forward_two_samples(self):
        """Two samples with different L produce correct output shapes after collate."""
        samples = [make_sample(L=10, M=4), make_sample(L=16, M=6)]
        batch   = collate_batch(samples)
        model   = UMAInverse(small_model_config())
        model.eval()
        with torch.no_grad():
            out = model(batch)

        assert out["logits"].shape == (2, 16, 21)
        assert out["pair_repr"].shape == (2, 22, 22, 32)  # 16 res + 6 lig
        assert torch.isfinite(out["logits"]).all()

    def test_loss_is_finite_and_positive(self):
        """Loss over designed residues should be a finite positive scalar."""
        batch = collate_batch([make_sample(L=12, M=5)])
        model = UMAInverse(small_model_config())
        model.eval()
        with torch.no_grad():
            loss = _loss_from_batch(model, batch)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_lightning_training_step(self):
        """UMAInverseLightningModule.training_step() returns finite loss."""
        module = UMAInverseLightningModule(
            model_config=small_model_config(),
            lr=1e-3,
            warmup_steps=10,
            T_max=100,
        )
        module.train()
        batch = collate_batch([make_sample(L=8, M=3)])
        loss  = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_lightning_training_step_deterministic(self):
        """Same input + same (epoch, batch_idx) → identical loss (deterministic decoding)."""
        import copy
        module = UMAInverseLightningModule(
            model_config=small_model_config(),
            lr=1e-3,
            warmup_steps=10,
            T_max=100,
        )
        module.train()
        batch = collate_batch([make_sample(L=8, M=3)])
        # Deep-copy before training_step mutates batch with decoding_order
        batch_copy = copy.deepcopy(batch)

        loss1 = module.training_step(batch,      batch_idx=5)
        loss2 = module.training_step(batch_copy, batch_idx=5)
        # With dropout=0 and seeded decoding order, identical inputs → identical loss
        assert torch.isclose(loss1, loss2, atol=1e-5)

    def test_gradient_flows_through_full_model(self):
        """Backward pass over the full model should produce finite, non-zero gradients."""
        model = UMAInverse(small_model_config())
        model.train()
        batch = collate_batch([make_sample(L=10, M=4)])
        loss  = _loss_from_batch(model, batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None,             f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(),   f"Non-finite gradient for {name}"


# ── Edge-case tests ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_ligand_atoms(self):
        """Structures with no ligand atoms must flow through without errors."""
        s = make_sample(L=10, M=0)
        batch = collate_batch([s])
        model = UMAInverse(small_model_config())
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (1, 10, 21)
        assert torch.isfinite(out["logits"]).all()

    def test_partial_design_mask(self):
        """Only designed residues contribute to loss; fixed positions are excluded."""
        s = make_sample(L=12, M=4)
        s["design_mask"][:6] = False  # fix first half
        batch = collate_batch([s])

        valid = batch["residue_mask"].bool() & batch["design_mask"].bool()
        assert valid[0, :6].sum() == 0, "Fixed residues should not be in loss mask"
        assert valid[0, 6:].sum() == 6, "Designed residues should all be in loss mask"

        model = UMAInverse(small_model_config())
        model.eval()
        with torch.no_grad():
            loss = _loss_from_batch(model, batch)
        assert torch.isfinite(loss)

    def test_padded_batch_no_nan(self):
        """Padded positions (mask=False) must not produce NaNs in any sample."""
        s1 = make_sample(L=8,  M=3, pdb_id="short")
        s2 = make_sample(L=20, M=6, pdb_id="long")
        batch = collate_batch([s1, s2])

        model = UMAInverse(small_model_config())
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert torch.isfinite(out["logits"]).all(), "NaN in padded batch logits"

    def test_single_residue(self):
        """A structure with L=1 and M=0 should not crash."""
        s = make_sample(L=1, M=0)
        batch = collate_batch([s])
        model = UMAInverse(small_model_config())
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (1, 1, 21)
        assert torch.isfinite(out["logits"]).all()

    def test_all_residues_fixed(self):
        """When design_mask is all False, forward pass must not crash."""
        s = make_sample(L=10, M=4)
        s["design_mask"][:] = False
        batch = collate_batch([s])
        model = UMAInverse(small_model_config())
        model.eval()
        with torch.no_grad():
            out = model(batch)
        # Forward pass itself should always be finite regardless of mask
        assert torch.isfinite(out["logits"]).all()

    @pytest.mark.parametrize("B,L,M", [(1, 1, 0), (2, 20, 10), (3, 50, 25), (1, 8, 25)])
    def test_various_sizes(self, B, L, M):
        """Model handles a range of (batch, length, ligand) sizes correctly."""
        samples = [make_sample(L=L, M=M) for _ in range(B)]
        batch   = collate_batch(samples)
        model   = UMAInverse(small_model_config())
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (B, L, 21)
        assert torch.isfinite(out["logits"]).all()


# ── Data pipeline robustness ──────────────────────────────────────────────────

class TestCollateRobustness:
    def test_sequence_padding_uses_token_20(self):
        """Padded sequence positions should be filled with 20 (unknown token)."""
        s1 = make_sample(L=5,  M=2)
        s2 = make_sample(L=10, M=2)
        batch = collate_batch([s1, s2])
        assert (batch["sequence"][0, 5:] == 20).all(), \
            "Padding should use token 20, not 0"

    def test_coord_padding_uses_zero(self):
        """Padded coordinate rows should be all zeros."""
        s1 = make_sample(L=4, M=2)
        s2 = make_sample(L=8, M=2)
        batch = collate_batch([s1, s2])
        assert (batch["residue_coords"][0, 4:] == 0).all()

    def test_mask_reflects_actual_lengths(self):
        """residue_mask should be True exactly for the original L positions."""
        s1 = make_sample(L=6,  M=2)
        s2 = make_sample(L=10, M=2)
        batch = collate_batch([s1, s2])
        assert batch["residue_mask"][0].sum() == 6
        assert batch["residue_mask"][1].sum() == 10

    def test_mixed_ligand_sizes(self):
        """Mix of M=0 and M>0 in same batch should not crash."""
        s1 = make_sample(L=8, M=0)
        s2 = make_sample(L=8, M=5)
        batch = collate_batch([s1, s2])
        assert batch["ligand_coords"].shape == (2, 5, 3)
        assert batch["ligand_mask"][0].sum() == 0
        assert batch["ligand_mask"][1].sum() == 5
