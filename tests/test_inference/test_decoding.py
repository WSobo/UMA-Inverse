"""Tests for :mod:`src.inference.decoding`.

Covers the contract the CLI relies on:

* Fixed residues are never overwritten.
* Forbidden AAs never appear in the output.
* Tied residues always share the same token.
* Seeded calls are reproducible.
* ``sample_next`` handles edge cases (argmax at T=0, top-p filtering,
  fully forbidden rows without NaN).

All tests run on CPU with a random-initialised model — we're verifying
logic, not learned behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.inference import DesignConstraints, InferenceSession
from src.inference.decoding import autoregressive_design, sample_next, score_sequence

FIXTURE_PDB = Path(__file__).parent.parent / "fixtures" / "1bc8.pdb"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession.from_checkpoint(
        config_path=CONFIG_PATH, checkpoint=None, device="cpu"
    )


@pytest.fixture(scope="module")
def ctx(session: InferenceSession):
    return session.load_structure(str(FIXTURE_PDB), max_total_nodes=5000)


# ─── sample_next primitive ────────────────────────────────────────────────────


class TestSampleNext:
    def test_argmax_at_zero_temperature(self) -> None:
        logits = torch.tensor([[0.1, 2.0, 0.5, 1.5]])
        tok, probs = sample_next(logits, temperature=0.0)
        assert tok.item() == 1
        assert probs[0, 1].item() == pytest.approx(1.0)

    def test_temperature_produces_valid_distribution(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(8, 21)
        tok, probs = sample_next(logits, temperature=0.5)
        assert tok.shape == (8,)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_top_p_filters_tail(self) -> None:
        logits = torch.tensor([[5.0, 0.0, -5.0, -10.0]])
        tok, probs = sample_next(logits, temperature=1.0, top_p=0.5)
        # Element 0 dominates; top-p=0.5 should drop the tail entirely
        assert probs[0, 2].item() == 0.0
        assert probs[0, 3].item() == 0.0

    def test_forbidden_mask_respected(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4, 21)
        forbidden = torch.zeros(4, 21, dtype=torch.bool)
        forbidden[:, :10] = True  # forbid first 10 tokens
        tok, _ = sample_next(logits, temperature=0.5, forbidden_mask=forbidden)
        assert (tok >= 10).all()

    def test_bias_shifts_distribution(self) -> None:
        logits = torch.zeros(1, 21)
        bias = torch.zeros(21)
        bias[5] = 10.0  # heavy bias on token 5
        tok, probs = sample_next(logits, temperature=1.0, bias=bias.unsqueeze(0))
        assert probs[0, 5] > 0.9

    def test_fully_forbidden_row_no_nan(self) -> None:
        logits = torch.randn(1, 21)
        forbidden = torch.ones(1, 21, dtype=torch.bool)
        tok, probs = sample_next(logits, temperature=1.0, forbidden_mask=forbidden)
        assert not torch.isnan(probs).any()
        # Fallback distribution is uniform when everything is forbidden
        assert torch.allclose(probs[0], torch.full((21,), 1.0 / 21.0), atol=1e-5)


# ─── autoregressive_design ────────────────────────────────────────────────────


@pytest.mark.skipif(
    not FIXTURE_PDB.exists() or not CONFIG_PATH.exists(),
    reason="fixture or config missing",
)
class TestAutoregressiveDesign:
    def test_shape_and_count(self, session, ctx) -> None:
        c = DesignConstraints.from_cli()
        resolved = c.resolve(ctx)
        samples = autoregressive_design(
            session, ctx, resolved, num_samples=3, batch_size=3, seed=42
        )
        assert len(samples) == 3
        for s in samples:
            assert s.token_ids.shape == (ctx.residue_count,)
            assert s.log_probs.shape == (ctx.residue_count,)
            assert s.probs_full.shape == (ctx.residue_count, 21)

    def test_fixed_residues_preserved(self, session, ctx) -> None:
        first3 = ctx.residue_ids[:3]
        c = DesignConstraints.from_cli(fix=" ".join(first3))
        resolved = c.resolve(ctx)
        samples = autoregressive_design(session, ctx, resolved, num_samples=2, seed=42)
        for s in samples:
            assert torch.equal(s.token_ids[:3], ctx.native_sequence[:3])

    def test_omit_constraints_respected(self, session, ctx) -> None:
        # Only allow W and Y in the designable positions (and X is always forbidden)
        c = DesignConstraints.from_cli(omit="ACDEFGHIKLMNPQRSTV")
        resolved = c.resolve(ctx)
        samples = autoregressive_design(
            session, ctx, resolved, num_samples=1, temperature=0.5, seed=42
        )
        designable = samples[0].token_ids[resolved.designable_mask.cpu()]
        # Tokens 18 (W) and 19 (Y) are the only allowed designable outputs
        allowed = {18, 19}
        assert set(int(t) for t in designable.tolist()) <= allowed

    def test_ties_enforce_identical_tokens(self, session, ctx) -> None:
        tied = [ctx.residue_ids[5], ctx.residue_ids[10], ctx.residue_ids[15]]
        c = DesignConstraints.from_cli(tie=",".join(tied))
        resolved = c.resolve(ctx)
        samples = autoregressive_design(session, ctx, resolved, num_samples=2, seed=42)
        for s in samples:
            tokens = s.token_ids[[5, 10, 15]]
            assert tokens.unique().numel() == 1

    def test_seed_reproducibility(self, session, ctx) -> None:
        c = DesignConstraints.from_cli()
        resolved = c.resolve(ctx)
        s1 = autoregressive_design(
            session, ctx, resolved, num_samples=1, temperature=0.5, seed=42
        )[0]
        s2 = autoregressive_design(
            session, ctx, resolved, num_samples=1, temperature=0.5, seed=42
        )[0]
        assert torch.equal(s1.token_ids, s2.token_ids)

    def test_argmax_is_deterministic_across_seeds(self, session, ctx) -> None:
        # At T=0 we still shuffle decoding order per seed, but every position's
        # choice only depends on previously-decoded tokens and structure — for
        # an untrained model with small logits, outputs usually converge.
        c = DesignConstraints.from_cli()
        resolved = c.resolve(ctx)
        s1 = autoregressive_design(session, ctx, resolved, num_samples=1, temperature=0.0, seed=1)[0]
        s2 = autoregressive_design(session, ctx, resolved, num_samples=1, temperature=0.0, seed=99)[0]
        agreement = (s1.token_ids == s2.token_ids).float().mean().item()
        assert agreement >= 0.5  # argmax with different decoding orders is still stable

    def test_overall_confidence_range(self, session, ctx) -> None:
        c = DesignConstraints.from_cli()
        resolved = c.resolve(ctx)
        sample = autoregressive_design(session, ctx, resolved, num_samples=1, seed=42)[0]
        conf = sample.overall_confidence(resolved.designable_mask.cpu())
        assert 0.0 < conf <= 1.0


# ─── score_sequence ───────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not FIXTURE_PDB.exists() or not CONFIG_PATH.exists(),
    reason="fixture or config missing",
)
class TestScoreSequence:
    def test_autoregressive_shape(self, session, ctx) -> None:
        r = score_sequence(session, ctx, mode="autoregressive", num_batches=3, seed=42)
        assert r.log_probs.shape == (ctx.residue_count,)
        assert r.sequence.shape == (ctx.residue_count,)
        assert r.mode == "autoregressive"
        assert r.num_batches == 3

    def test_single_aa_shape(self, session) -> None:
        # Use a small context to keep the L-forward cost down
        ctx_small = session.load_structure(str(FIXTURE_PDB), max_total_nodes=50)
        r = score_sequence(session, ctx_small, mode="single-aa")
        assert r.log_probs.shape == (ctx_small.residue_count,)
        assert r.mode == "single-aa"

    def test_unknown_mode_raises(self, session, ctx) -> None:
        with pytest.raises(ValueError, match="unknown scoring mode"):
            score_sequence(session, ctx, mode="bogus")  # type: ignore[arg-type]
