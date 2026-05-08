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


def _v3_sample(L: int, M: int, name: str, sc_per_res: int = 4) -> dict[str, torch.Tensor]:
    """Build a v3-shaped sample. Includes synthetic sidechain heavy atoms
    (sc_per_res atoms per residue) so the geometric aug can pull them.
    """
    K = L * sc_per_res
    sc_residue_idx = torch.arange(L).repeat_interleave(sc_per_res)
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
        "sidechain_coords": torch.randn(K, 3),
        "sidechain_atomic_numbers": torch.randint(1, 19, (K,)),
        "sidechain_residue_idx": sc_residue_idx,
        "pdb_id": name,
    }


def test_collate_v3_frame_angles_pad_lm():
    """Frame-angle tensor [L, M, 4] should pad to [B, max_L, max_M, 4].

    Sidechain_* keys are consumed by the per-sample aug or dropped before
    collate, so they must NOT appear in the batched output.
    """
    a = _v3_sample(L=5, M=3, name="A")
    b = _v3_sample(L=8, M=1, name="B")
    # Strip sidechain_* the way Dataset.__getitem__ does after aug/drop.
    for s in (a, b):
        for k in ("sidechain_coords", "sidechain_atomic_numbers", "sidechain_residue_idx"):
            s.pop(k, None)
    batch = collate_batch([a, b])

    assert batch["residue_ligand_frame_angles"].shape == (2, 8, 3, 4)
    # Sample B has M=1, so the padded slots [:, :, 1:, :] for that row
    # should be zero.
    assert batch["residue_ligand_frame_angles"][1, :, 1:, :].abs().sum() == 0
    # Sample A has L=5, so rows [5:] should be zero too.
    assert batch["residue_ligand_frame_angles"][0, 5:, :, :].abs().sum() == 0
    # Sidechain tensors must not be batched — collate has no schema for them.
    assert "sidechain_coords" not in batch
    assert "sidechain_residue_idx" not in batch


def test_apply_sidechain_context_aug_appends_atoms():
    """v3 phase 5 — augmentation appends sidechain atoms of chosen residues
    to ligand_coords / ligand_atomic_numbers and recomputes frame angles.
    """
    L, M, sc_per_res = 100, 4, 5
    item = _v3_sample(L=L, M=M, name="X", sc_per_res=sc_per_res)
    item["design_mask"] = torch.ones(L, dtype=torch.bool)

    rng = random.Random(0)
    out = _apply_sidechain_context_aug(item, rate=0.10, rng=rng)

    # 10 residues × 5 atoms each = 50 sidechain atoms appended.
    expected_appended = 10 * sc_per_res
    assert out["ligand_coords"].shape[0] == M + expected_appended
    assert out["ligand_atomic_numbers"].shape[0] == M + expected_appended
    assert out["ligand_mask"].shape[0] == M + expected_appended
    assert out["ligand_mask"].all()
    # Frame angles must re-cover the new ligand size in the M dim.
    assert out["residue_ligand_frame_angles"].shape == (L, M + expected_appended, 4)
    # Sidechain_* keys are consumed by the aug.
    assert "sidechain_coords" not in out
    assert "sidechain_atomic_numbers" not in out
    assert "sidechain_residue_idx" not in out


def test_apply_sidechain_context_aug_zero_rate_noop():
    """rate=0.0 must leave ligand tensors untouched."""
    item = _v3_sample(L=8, M=3, name="X")
    rng = random.Random(0)
    pre_M = item["ligand_coords"].shape[0]
    out = _apply_sidechain_context_aug(item, rate=0.0, rng=rng)
    assert out is item
    assert out["ligand_coords"].shape[0] == pre_M


def test_apply_sidechain_context_aug_only_designable():
    """Sampling must only pull sidechain atoms from designable residues."""
    L, sc_per_res = 20, 3
    item = _v3_sample(L=L, M=2, name="X", sc_per_res=sc_per_res)
    # First half designable, second half not.
    dm = torch.zeros(L, dtype=torch.bool)
    dm[:10] = True
    item["design_mask"] = dm
    pre_M = item["ligand_coords"].shape[0]

    rng = random.Random(0)
    out = _apply_sidechain_context_aug(item, rate=1.0, rng=rng)
    appended = out["ligand_coords"].shape[0] - pre_M
    # rate=1.0 + 10 designable + 3 atoms/residue = 30 atoms appended.
    assert appended == 10 * sc_per_res
