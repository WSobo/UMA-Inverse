from omegaconf import OmegaConf

from src.models.uma_inverse import UMAInverse


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    model = UMAInverse(OmegaConf.to_container(cfg.model, resolve=True))
    print("Model instantiated:", model.__class__.__name__)
