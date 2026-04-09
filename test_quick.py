import torch
from omegaconf import OmegaConf

from src.models.uma_inverse import UMAInverse


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    model = UMAInverse(OmegaConf.to_container(cfg.model, resolve=True))

    batch = {
        "residue_coords": torch.randn(1, 16, 3),
        "residue_features": torch.randn(1, 16, 6),
        "residue_mask": torch.ones(1, 16, dtype=torch.bool),
        "sequence": torch.randint(0, 20, (1, 16)),
        "design_mask": torch.ones(1, 16, dtype=torch.bool),
        "ligand_coords": torch.randn(1, 5, 3),
        "ligand_features": torch.randn(1, 5, 6),
        "ligand_mask": torch.ones(1, 5, dtype=torch.bool),
    }

    with torch.no_grad():
        out = model(batch)

    print("Quick forward ok. Logits shape:", tuple(out["logits"].shape))
