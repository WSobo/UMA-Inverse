import torch

from src.data.datamodule import collate_batch


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
