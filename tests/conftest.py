"""pytest configuration and shared fixtures for UMA-Inverse tests."""
import os
import sys

import pytest
import torch

# Ensure project root is on sys.path before any src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Determinism ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def seed_everything():
    """Seed all RNGs before each test for deterministic results."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


# ── Sample-building helpers ───────────────────────────────────────────────────

def make_sample(L: int = 12, M: int = 5, pdb_id: str = "TEST") -> dict:
    """Build a single un-batched sample dict (as returned by UMAInverseDataset)."""
    sample = {
        "residue_coords":   torch.randn(L, 3),
        "residue_features": torch.randn(L, 6),
        "residue_mask":     torch.ones(L,  dtype=torch.bool),
        "sequence":         torch.randint(0, 20, (L,)),
        "design_mask":      torch.ones(L,  dtype=torch.bool),
        "ligand_coords":    torch.randn(M, 3) if M > 0 else torch.zeros(0, 3),
        "ligand_features":  torch.randn(M, 6) if M > 0 else torch.zeros(0, 6),
        "ligand_mask":      torch.ones(M, dtype=torch.bool) if M > 0 else torch.zeros(0, dtype=torch.bool),
        "pdb_id":           pdb_id,
    }
    return sample


def small_model_config(**overrides) -> dict:
    """Return a minimal model config for fast unit tests."""
    cfg = {
        "residue_input_dim":    6,
        "ligand_input_dim":     6,
        "node_dim":             32,
        "pair_dim":             32,
        "pair_hidden_dim":      32,
        "num_pairmixer_blocks": 2,
        "pair_transition_mult": 2,
        "num_rbf":              8,
        "max_distance":         12.0,
        "dropout":              0.0,
        "gradient_checkpointing": False,
        "thermal_noise_std":    0.0,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def sample_batch():
    """A two-sample collated batch with different sequence lengths."""
    from src.data.datamodule import collate_batch
    samples = [make_sample(L=10, M=4, pdb_id="A"), make_sample(L=16, M=6, pdb_id="B")]
    return collate_batch(samples)


@pytest.fixture
def small_config():
    return small_model_config()
