import random

import torch

from src.data.datamodule import _apply_sidechain_context_aug, collate_batch


def test_collate_batch_padding_shapes():
    sample_a = {
        "residue_coords": torch.randn(5, 3),
        "residue_features": torch.randn(5, 6),
        "residue_mask": torch.ones(5, dtype=torch.bool),
        "sequence": torch.randint(0, 20, (5,)),
        "design_mask": torch.ones(5, dtype=torch.bool),
        "ligand_coords": torch.randn(3, 3),
        "ligand_features": torch.randn(3, 6),
        "ligand_mask": torch.ones(3, dtype=torch.bool),
        "pdb_id": "A",
    }

    sample_b = {
        "residue_coords": torch.randn(8, 3),
        "residue_features": torch.randn(8, 6),
        "residue_mask": torch.ones(8, dtype=torch.bool),
        "sequence": torch.randint(0, 20, (8,)),
        "design_mask": torch.ones(8, dtype=torch.bool),
        "ligand_coords": torch.randn(1, 3),
        "ligand_features": torch.randn(1, 6),
        "ligand_mask": torch.ones(1, dtype=torch.bool),
        "pdb_id": "B",
    }

    batch = collate_batch([sample_a, sample_b])

    assert batch["residue_coords"].shape == (2, 8, 3)
    assert batch["residue_features"].shape == (2, 8, 6)
    assert batch["ligand_coords"].shape == (2, 3, 3)
    assert batch["ligand_features"].shape == (2, 3, 6)
    assert batch["sequence"].shape == (2, 8)


def _v3_sample(L: int, M: int, name: str) -> dict[str, torch.Tensor]:
    return {
        "residue_coords": torch.randn(L, 3),
        "residue_features": torch.randn(L, 6),
        "residue_mask": torch.ones(L, dtype=torch.bool),
        "sequence": torch.randint(0, 20, (L,)),
        "design_mask": torch.ones(L, dtype=torch.bool),
        "ligand_coords": torch.randn(M, 3),
        "ligand_atomic_numbers": torch.randint(1, 119, (M,)),
        "ligand_mask": torch.ones(M, dtype=torch.bool),
        "residue_backbone_coords": torch.randn(L, 4, 3),
        "residue_ligand_frame_angles": torch.randn(L, M, 4),
        "sidechain_context_mask": torch.zeros(L, dtype=torch.bool),
        "pdb_id": name,
    }


def test_collate_v3_frame_angles_pad_lm():
    """Frame-angle tensor [L, M, 4] should pad to [B, max_L, max_M, 4]."""
    a = _v3_sample(L=5, M=3, name="A")
    b = _v3_sample(L=8, M=1, name="B")
    batch = collate_batch([a, b])

    assert batch["residue_ligand_frame_angles"].shape == (2, 8, 3, 4)
    # Sample B has M=1, so the padded slots [:, :, 1:, :] for that row
    # should be zero.
    assert batch["residue_ligand_frame_angles"][1, :, 1:, :].abs().sum() == 0
    # Sample A has L=5, so rows [5:] should be zero too.
    assert batch["residue_ligand_frame_angles"][0, 5:, :, :].abs().sum() == 0


def test_collate_v3_sidechain_mask_pad():
    """sidechain_context_mask [L] should pad to [B, max_L]."""
    a = _v3_sample(L=5, M=2, name="A")
    a["sidechain_context_mask"] = torch.tensor([True, False, True, False, True])
    b = _v3_sample(L=8, M=2, name="B")
    batch = collate_batch([a, b])

    assert batch["sidechain_context_mask"].shape == (2, 8)
    assert batch["sidechain_context_mask"].dtype == torch.bool
    # First sample's mask preserved, padded slots False.
    assert batch["sidechain_context_mask"][0, :5].tolist() == [True, False, True, False, True]
    assert not batch["sidechain_context_mask"][0, 5:].any()
    # Second sample's mask is all False (default), all-False after pad.
    assert not batch["sidechain_context_mask"][1].any()


def test_apply_sidechain_context_aug_rate_respected():
    """Augmentation should flag approximately `rate * len(designable)` positions.

    Uses a deterministic RNG so the test is stable.
    """
    L = 100
    item = _v3_sample(L=L, M=4, name="X")
    # All residues designable.
    item["design_mask"] = torch.ones(L, dtype=torch.bool)

    rng = random.Random(0)
    out = _apply_sidechain_context_aug(item, rate=0.10, rng=rng)
    n_flagged = int(out["sidechain_context_mask"].sum().item())
    # round(100 * 0.10) = 10
    assert n_flagged == 10
    assert out["design_mask"].equal(item["design_mask"])  # design_mask untouched


def test_apply_sidechain_context_aug_zero_rate_noop():
    """rate=0.0 should leave the item unchanged."""
    item = _v3_sample(L=8, M=3, name="X")
    rng = random.Random(0)
    out = _apply_sidechain_context_aug(item, rate=0.0, rng=rng)
    # No new key added.
    assert out is item or out.get("sidechain_context_mask") is None


def test_apply_sidechain_context_aug_only_designable():
    """Sampling should only flag positions where design_mask is True."""
    L = 20
    item = _v3_sample(L=L, M=2, name="X")
    # First half designable, second half not.
    dm = torch.zeros(L, dtype=torch.bool)
    dm[:10] = True
    item["design_mask"] = dm

    rng = random.Random(0)
    out = _apply_sidechain_context_aug(item, rate=1.0, rng=rng)
    flagged = out["sidechain_context_mask"]
    # Every flagged index must be in the designable half.
    assert flagged[:10].all()
    assert not flagged[10:].any()
