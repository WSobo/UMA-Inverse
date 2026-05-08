"""Tests for the UMAInverse end-to-end model."""

import pytest
import torch

from src.models.uma_inverse import RBFEmbedding, UMAInverse

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _small_config(**overrides):
    cfg = {
        "residue_input_dim": 6,
        "ligand_input_dim": 6,
        "node_dim": 32,
        "pair_dim": 32,
        "pair_hidden_dim": 32,
        "num_pairmixer_blocks": 2,
        "pair_transition_mult": 2,
        "num_rbf": 8,
        "max_distance": 12.0,
        "dropout": 0.0,
        "gradient_checkpointing": False,
        "thermal_noise_std": 0.0,
    }
    cfg.update(overrides)
    return cfg


def _make_batch(B=1, L=16, M=5, device="cpu"):
    return {
        "residue_coords": torch.randn(B, L, 3, device=device),
        "residue_features": torch.randn(B, L, 6, device=device),
        "residue_mask": torch.ones(B, L, dtype=torch.bool, device=device),
        "sequence": torch.randint(0, 20, (B, L), device=device),
        "design_mask": torch.ones(B, L, dtype=torch.bool, device=device),
        "ligand_coords": torch.randn(B, M, 3, device=device),
        "ligand_features": torch.randn(B, M, 6, device=device),
        "ligand_mask": torch.ones(B, M, dtype=torch.bool, device=device),
    }


# ──────────────────────────────────────────────────────────────────────────────
# RBFEmbedding
# ──────────────────────────────────────────────────────────────────────────────

def test_rbf_shape():
    rbf = RBFEmbedding(num_rbf=16, max_distance=20.0)
    dist = torch.rand(2, 10, 10) * 20.0
    out = rbf(dist)
    assert out.shape == (2, 10, 10, 16)
    assert torch.isfinite(out).all()


def test_rbf_range():
    """All RBF values should be in (0, 1]."""
    rbf = RBFEmbedding(num_rbf=8, max_distance=10.0)
    dist = torch.linspace(0, 10, 50)
    out = rbf(dist)
    assert (out >= 0).all() and (out <= 1 + 1e-5).all()


# ──────────────────────────────────────────────────────────────────────────────
# UMAInverse forward
# ──────────────────────────────────────────────────────────────────────────────

def test_uma_inverse_forward_shape():
    torch.manual_seed(1)
    model = UMAInverse(_small_config())
    batch = _make_batch(B=2, L=20, M=8)
    out = model(batch)
    assert out["logits"].shape == (2, 20, 21)
    assert out["pair_repr"].shape == (2, 28, 28, 32)


def test_uma_inverse_no_ligand():
    """Model must handle zero-atom ligand gracefully."""
    model = UMAInverse(_small_config())
    batch = _make_batch(B=1, L=10, M=0)
    batch["ligand_coords"] = torch.zeros(1, 0, 3)
    batch["ligand_features"] = torch.zeros(1, 0, 6)
    batch["ligand_mask"] = torch.zeros(1, 0, dtype=torch.bool)
    out = model(batch)
    assert out["logits"].shape == (1, 10, 21)
    assert torch.isfinite(out["logits"]).all()


def test_uma_inverse_no_sequence_gives_zero_ar_context():
    """Without a sequence key the AR context should be all zeros."""
    model = UMAInverse(_small_config())
    batch = _make_batch(B=1, L=8, M=3)
    batch.pop("sequence")
    out = model(batch)
    assert out["logits"].shape == (1, 8, 21)
    assert torch.isfinite(out["logits"]).all()


def test_uma_inverse_padded_mask():
    """Padded positions (mask=0) should not cause NaNs."""
    model = UMAInverse(_small_config())
    batch = _make_batch(B=2, L=20, M=5)
    # Pad the second sample to L=12 (rows 12-19 are padding)
    batch["residue_mask"][1, 12:] = False
    out = model(batch)
    assert torch.isfinite(out["logits"]).all()


def test_uma_inverse_gradient_flow():
    """Backward pass should produce finite, non-zero gradients."""
    model = UMAInverse(_small_config())
    batch = _make_batch(B=1, L=10, M=4)
    out = model(batch)
    loss = out["logits"].sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


def test_uma_inverse_logits_sum_to_nonzero():
    """Logits should not be all-zero (would indicate dead initialisation)."""
    torch.manual_seed(99)
    model = UMAInverse(_small_config())
    batch = _make_batch(B=1, L=8, M=3)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].abs().sum() > 0


@pytest.mark.parametrize("B,L,M", [(1, 8, 0), (2, 20, 10), (1, 50, 25)])
def test_uma_inverse_various_sizes(B, L, M):
    model = UMAInverse(_small_config())
    batch = _make_batch(B=B, L=L, M=M)
    if M == 0:
        batch["ligand_coords"] = torch.zeros(B, 0, 3)
        batch["ligand_features"] = torch.zeros(B, 0, 6)
        batch["ligand_mask"] = torch.zeros(B, 0, dtype=torch.bool)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (B, L, 21)
    assert torch.isfinite(out["logits"]).all()


def test_uma_inverse_thermal_noise_train_only():
    """Thermal noise should be applied in train mode and not in eval mode."""
    model = UMAInverse(_small_config(thermal_noise_std=10.0))
    batch = _make_batch(B=1, L=8, M=3)

    model.train()
    out_train = model(batch)

    model.eval()
    with torch.no_grad():
        out_eval = model(batch)

    # With large noise, train and eval logits will differ
    assert not torch.allclose(out_train["logits"].detach(), out_eval["logits"], atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# v3 feature flags
# ──────────────────────────────────────────────────────────────────────────────

def _v3_batch(B=1, L=12, M=5):
    batch = _make_batch(B=B, L=L, M=M)
    batch["residue_backbone_coords"] = torch.randn(B, L, 4, 3)
    batch["residue_ligand_frame_angles"] = torch.randn(B, L, M, 4)
    batch["sidechain_context_mask"] = torch.zeros(B, L, dtype=torch.bool)
    batch["ligand_atomic_numbers"] = torch.randint(1, 119, (B, M))
    batch.pop("ligand_features", None)
    return batch


def test_v3_pair_distance_atoms_ligand_backbone_full():
    """Feature 1: 5-atom × ligand-atom distances replace the [L,M]/[M,L] block."""
    cfg = _small_config(
        pair_distance_atoms="backbone_full",
        pair_distance_atoms_ligand="backbone_full",
        ligand_featurizer="atomic_number_embedding",
    )
    model = UMAInverse(cfg)
    out = model(_v3_batch())
    assert torch.isfinite(out["logits"]).all()


def test_v3_frame_relative_angles():
    """Feature 2: frame-relative angles project into the [L,M] pair block."""
    cfg = _small_config(
        pair_distance_atoms="backbone_full",
        frame_relative_angles=True,
        ligand_featurizer="atomic_number_embedding",
    )
    model = UMAInverse(cfg)
    out = model(_v3_batch())
    assert torch.isfinite(out["logits"]).all()


def test_v3_intra_ligand_multidist():
    """Feature 3: ligand_local_env enriches ligand node embeddings."""
    cfg = _small_config(
        intra_ligand_multidist=True,
        ligand_featurizer="atomic_number_embedding",
    )
    model = UMAInverse(cfg)
    batch = _v3_batch(M=6)  # need M >= num_neighbors+1=4
    out = model(batch)
    assert torch.isfinite(out["logits"]).all()
    # ligand_local_env should produce non-zero contribution
    env = model.ligand_local_env(batch["ligand_coords"], batch["ligand_mask"])
    assert env.abs().sum() > 0


def test_v3_coord_noise_train_only():
    """Feature 4: raw-coord noise active in train, suppressed in eval."""
    cfg = _small_config(
        coord_noise_std=2.0,  # large enough to perturb output substantially
        ligand_featurizer="atomic_number_embedding",
    )
    model = UMAInverse(cfg)
    batch = _v3_batch()

    torch.manual_seed(0)
    model.train()
    out_a = model(batch)["logits"].detach()
    torch.manual_seed(1)
    out_b = model(batch)["logits"].detach()
    assert not torch.allclose(out_a, out_b, atol=1e-3)

    model.eval()
    with torch.no_grad():
        out_eval_a = model(batch)["logits"]
        out_eval_b = model(batch)["logits"]
    assert torch.allclose(out_eval_a, out_eval_b)


def test_v3_sidechain_context_changes_logits():
    """Feature 5: flagged positions become bidirectionally visible in AR context.

    Setting sidechain_context_mask should produce different logits than not
    setting it, because the causal attention now sees more positions.
    """
    cfg = _small_config(ligand_featurizer="atomic_number_embedding")
    model = UMAInverse(cfg)
    model.eval()
    batch = _v3_batch(L=10, M=4)

    with torch.no_grad():
        out_no_aug = model(batch)["logits"]

    # Flag half the positions as visible in AR context.
    batch["sidechain_context_mask"] = torch.zeros(1, 10, dtype=torch.bool)
    batch["sidechain_context_mask"][0, ::2] = True
    with torch.no_grad():
        out_with_aug = model(batch)["logits"]

    assert not torch.allclose(out_no_aug, out_with_aug, atol=1e-3)


def test_v3_all_flags_on_gradient_flow():
    """All 5 v3 features active simultaneously: forward + backward, no NaN."""
    cfg = _small_config(
        pair_distance_atoms="backbone_full",
        pair_distance_atoms_ligand="backbone_full",
        frame_relative_angles=True,
        intra_ligand_multidist=True,
        coord_noise_std=0.1,
        ligand_featurizer="atomic_number_embedding",
    )
    model = UMAInverse(cfg)
    model.train()
    batch = _v3_batch(B=2, L=14, M=6)
    out = model(batch)
    loss = out["logits"].mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is None or torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_v3_off_equals_v2_baseline():
    """Regression guard: v3 codebase with all v3 flags OFF must produce
    numerically identical logits to v2 for the same batch and seed.
    """
    common = _small_config(
        pair_distance_atoms="backbone_full",
        ligand_featurizer="atomic_number_embedding",
    )
    # v3 codebase, all v3 flags explicitly OFF (their defaults).
    v3_cfg = dict(common)
    v3_cfg.update(
        pair_distance_atoms_ligand="anchor_only",
        frame_relative_angles=False,
        intra_ligand_multidist=False,
        coord_noise_std=0.0,
    )

    torch.manual_seed(42)
    model_v2 = UMAInverse(common)
    torch.manual_seed(42)
    model_v3 = UMAInverse(v3_cfg)

    batch = _v3_batch(B=2, L=12, M=5)
    # v2 only consumes residue_backbone_coords + the standard v2 keys, so
    # passing the v3-only keys is harmless — it just ignores them.
    model_v2.eval()
    model_v3.eval()
    with torch.no_grad():
        out_v2 = model_v2(batch)["logits"]
        out_v3 = model_v3(batch)["logits"]
    assert torch.allclose(out_v2, out_v3, atol=1e-6, rtol=1e-5), (
        "v3 with all flags off must match v2 logits exactly"
    )
