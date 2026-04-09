import torch

from src.models.uma_inverse import UMAInverse


def test_uma_inverse_forward_shape():
    torch.manual_seed(1)
    model = UMAInverse(
        {
            "residue_input_dim": 6,
            "ligand_input_dim": 6,
            "node_dim": 64,
            "pair_dim": 64,
            "pair_hidden_dim": 64,
            "num_pairmixer_blocks": 2,
            "pair_transition_mult": 2,
            "num_rbf": 16,
            "dropout": 0.0,
            "gradient_checkpointing": False,
        }
    )

    batch = {
        "residue_coords": torch.randn(2, 20, 3),
        "residue_features": torch.randn(2, 20, 6),
        "residue_mask": torch.ones(2, 20, dtype=torch.bool),
        "sequence": torch.randint(0, 20, (2, 20)),
        "design_mask": torch.ones(2, 20, dtype=torch.bool),
        "ligand_coords": torch.randn(2, 8, 3),
        "ligand_features": torch.randn(2, 8, 6),
        "ligand_mask": torch.ones(2, 8, dtype=torch.bool),
    }

    out = model(batch)

    assert out["logits"].shape == (2, 20, 21)
    assert out["pair_repr"].shape == (2, 28, 28, 64)
