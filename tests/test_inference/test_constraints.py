"""Tests for :mod:`src.inference.constraints`.

These exercise every user-facing parser in isolation (so error messages stay
sharp) plus the end-to-end ``DesignConstraints.resolve`` path against a
realistic :class:`StructureContext`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.inference import DesignConstraints, InferenceSession, parse_residue_selection
from src.inference.constraints import (
    AA_TO_ID,
    ConstraintError,
    as_aa_letters,
    as_token_ids,
    load_per_residue_bias,
    load_per_residue_omit,
    parse_aa_bias,
    parse_aa_omit,
    parse_tie_groups,
)

FIXTURE_PDB = Path(__file__).parent.parent / "fixtures" / "1bc8.pdb"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


# ─── Residue selection parser ─────────────────────────────────────────────────


class TestParseResidueSelection:
    def test_empty_is_empty_list(self) -> None:
        assert parse_residue_selection(None) == []
        assert parse_residue_selection("") == []
        assert parse_residue_selection("  ") == []

    def test_space_and_comma_separators(self) -> None:
        assert parse_residue_selection("A1 A2 A3") == ["A1", "A2", "A3"]
        assert parse_residue_selection("A1,A2,A3") == ["A1", "A2", "A3"]
        assert parse_residue_selection("A1, A2  A3") == ["A1", "A2", "A3"]

    def test_dedup_preserves_order(self) -> None:
        assert parse_residue_selection("A1 A2 A1 A3") == ["A1", "A2", "A3"]

    def test_insertion_codes(self) -> None:
        assert parse_residue_selection("B42A B42B") == ["B42A", "B42B"]

    def test_negative_resnum(self) -> None:
        assert parse_residue_selection("C-5") == ["C-5"]

    def test_invalid_token_raises(self) -> None:
        with pytest.raises(ConstraintError, match="invalid residue id"):
            parse_residue_selection("1A")
        with pytest.raises(ConstraintError, match="invalid residue id"):
            parse_residue_selection("AA1")


# ─── Bias / omit parsers ──────────────────────────────────────────────────────


class TestParseAaBias:
    def test_basic_pairs(self) -> None:
        out = parse_aa_bias("W:3.0,A:-1.0")
        assert out == {AA_TO_ID["W"]: 3.0, AA_TO_ID["A"]: -1.0}

    def test_empty(self) -> None:
        assert parse_aa_bias(None) == {}
        assert parse_aa_bias("") == {}

    def test_case_insensitive(self) -> None:
        assert parse_aa_bias("w:1.0") == {AA_TO_ID["W"]: 1.0}

    def test_invalid_format(self) -> None:
        with pytest.raises(ConstraintError, match="invalid bias pair"):
            parse_aa_bias("W3.0")

    def test_invalid_aa(self) -> None:
        with pytest.raises(ConstraintError, match="invalid AA"):
            parse_aa_bias("Z:1.0")

    def test_x_not_allowed(self) -> None:
        # X is the unknown/non-standard token — must be rejected for bias
        with pytest.raises(ConstraintError):
            parse_aa_bias("X:1.0")


class TestParseAaOmit:
    def test_concatenated(self) -> None:
        ids = parse_aa_omit("CDFG")
        assert ids == {AA_TO_ID[c] for c in "CDFG"}

    def test_comma_separated(self) -> None:
        assert parse_aa_omit("C,D,F,G") == parse_aa_omit("CDFG")

    def test_invalid_letter_raises(self) -> None:
        with pytest.raises(ConstraintError, match="invalid AA"):
            parse_aa_omit("XZ")


class TestParseTieGroups:
    def test_single_group_default_weights(self) -> None:
        result = parse_tie_groups("A1,A2,A3")
        assert len(result) == 1
        members, weights = result[0]
        assert members == ["A1", "A2", "A3"]
        assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_multiple_groups(self) -> None:
        result = parse_tie_groups("A1,A2|B5,B6,B7")
        assert len(result) == 2
        assert result[0][0] == ["A1", "A2"]
        assert result[1][0] == ["B5", "B6", "B7"]

    def test_explicit_weights(self) -> None:
        result = parse_tie_groups("A1,A2|B1,B2,B3", weights="0.3,0.7|0.5,0.5,0.5")
        assert result[0][1] == [0.3, 0.7]
        assert result[1][1] == [0.5, 0.5, 0.5]

    def test_weight_count_mismatch(self) -> None:
        with pytest.raises(ConstraintError, match="tie group 0 has 2 residues but 3 weights"):
            parse_tie_groups("A1,A2", weights="0.3,0.3,0.4")

    def test_group_count_mismatch(self) -> None:
        with pytest.raises(ConstraintError, match="tie-weights group count"):
            parse_tie_groups("A1,A2|B1,B2", weights="0.5,0.5")


# ─── JSON loaders ─────────────────────────────────────────────────────────────


def _write_json(tmp_path: Path, name: str, body: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(body))
    return path


class TestPerResidueBiasFile:
    def test_round_trip(self, tmp_path: Path) -> None:
        spec = {"A23": {"W": 3.0, "A": -1.0}, "B42": {"P": 10.0}}
        path = _write_json(tmp_path, "bias.json", spec)
        loaded = load_per_residue_bias(path)
        assert loaded["A23"][AA_TO_ID["W"]] == 3.0
        assert loaded["A23"][AA_TO_ID["A"]] == -1.0
        assert loaded["B42"][AA_TO_ID["P"]] == 10.0

    def test_invalid_aa_raises(self, tmp_path: Path) -> None:
        path = _write_json(tmp_path, "bias.json", {"A1": {"Z": 1.0}})
        with pytest.raises(ConstraintError, match="invalid AA"):
            load_per_residue_bias(path)

    def test_invalid_residue_id_raises(self, tmp_path: Path) -> None:
        path = _write_json(tmp_path, "bias.json", {"not-a-residue": {"A": 1.0}})
        with pytest.raises(ConstraintError, match="invalid residue id"):
            load_per_residue_bias(path)


class TestPerResidueOmitFile:
    def test_round_trip(self, tmp_path: Path) -> None:
        spec = {"A23": "CDFG", "B42": "WY"}
        path = _write_json(tmp_path, "omit.json", spec)
        loaded = load_per_residue_omit(path)
        assert loaded["A23"] == {AA_TO_ID[c] for c in "CDFG"}
        assert loaded["B42"] == {AA_TO_ID["W"], AA_TO_ID["Y"]}


# ─── Token <-> letter helpers ─────────────────────────────────────────────────


class TestTokenConversions:
    def test_round_trip(self) -> None:
        original = "ACDEFGHIKLMNPQRSTVWY"
        ids = as_token_ids(original)
        back = as_aa_letters(ids)
        assert back == original

    def test_as_token_ids_rejects_unknown(self) -> None:
        with pytest.raises(ConstraintError, match="unknown AA letter"):
            as_token_ids(["Z"])


# ─── from_cli composite ───────────────────────────────────────────────────────


class TestFromCli:
    def test_composes_all_fields(self) -> None:
        c = DesignConstraints.from_cli(
            fix="A1,A2",
            redesign="A3",
            design_chains="A,B",
            bias="W:3.0",
            omit="CDFG",
            tie="A4,A5|A6,A7",
        )
        assert c.fix == {"A1", "A2"}
        assert c.redesign == {"A3"}
        assert c.design_chains == {"A", "B"}
        assert c.bias == {AA_TO_ID["W"]: 3.0}
        assert len(c.omit) == 4
        assert len(c.ties) == 2

    def test_fix_redesign_overlap_raises(self) -> None:
        with pytest.raises(ConstraintError, match="appear in both"):
            DesignConstraints.from_cli(fix="A1", redesign="A1")

    def test_invalid_chain_raises(self) -> None:
        with pytest.raises(ConstraintError, match="invalid chain"):
            DesignConstraints.from_cli(design_chains="AB")


# ─── resolve() against a real structure ───────────────────────────────────────


@pytest.mark.skipif(
    not FIXTURE_PDB.exists() or not CONFIG_PATH.exists(),
    reason="fixture PDB or config missing",
)
class TestResolveAgainstStructure:
    @pytest.fixture(scope="class")
    def ctx(self):
        session = InferenceSession.from_checkpoint(
            config_path=CONFIG_PATH, checkpoint=None, device="cpu"
        )
        return session.load_structure(str(FIXTURE_PDB), max_total_nodes=5000)

    def test_resolve_basic(self, ctx) -> None:
        first_three = ctx.residue_ids[:3]
        c = DesignConstraints.from_cli(fix=" ".join(first_three), bias="W:3.0")
        resolved = c.resolve(ctx)
        fixed_indices = resolved.fixed_mask.nonzero().flatten().tolist()
        assert fixed_indices == [0, 1, 2]
        # Fixed residues are excluded from the designable set
        for idx in fixed_indices:
            assert not bool(resolved.designable_mask[idx])

    def test_resolve_design_chains(self, ctx) -> None:
        chains = sorted(set(ctx.chain_ids))
        first_chain = chains[0]
        c = DesignConstraints.from_cli(design_chains=first_chain)
        resolved = c.resolve(ctx)
        for cid, is_designable in zip(ctx.chain_ids, resolved.designable_mask.tolist()):
            if cid == first_chain:
                assert is_designable
            else:
                assert not is_designable

    def test_resolve_missing_residue_raises(self, ctx) -> None:
        c = DesignConstraints.from_cli(fix="Z999")
        with pytest.raises(ConstraintError, match="not found in"):
            c.resolve(ctx)

    def test_resolve_unknown_chain_in_design_chains(self, ctx) -> None:
        c = DesignConstraints.from_cli(design_chains="Z")
        with pytest.raises(ConstraintError, match="references chains not present"):
            c.resolve(ctx)

    def test_x_token_always_forbidden(self, ctx) -> None:
        c = DesignConstraints.from_cli()
        resolved = c.resolve(ctx)
        assert bool(resolved.omit_global[20])  # X = 20

    def test_bias_file_resolves(self, ctx, tmp_path: Path) -> None:
        spec = {ctx.residue_ids[0]: {"W": 5.0}}
        path = tmp_path / "bias.json"
        path.write_text(json.dumps(spec))
        c = DesignConstraints.from_cli(bias_file=str(path))
        resolved = c.resolve(ctx)
        assert resolved.bias_per_residue[0, AA_TO_ID["W"]].item() == 5.0

    def test_resolve_tensors_on_context_device(self, ctx) -> None:
        c = DesignConstraints.from_cli(fix=ctx.residue_ids[0])
        resolved = c.resolve(ctx)
        assert resolved.fixed_mask.device == ctx.device
        assert resolved.bias_global.device == ctx.device

    def test_logit_bias_and_forbidden_mask_helpers(self, ctx) -> None:
        c = DesignConstraints.from_cli(bias="W:3.0", omit="CDFG")
        resolved = c.resolve(ctx)
        # Any position's logit_bias should pick up the global W=3.0
        lb = resolved.logit_bias(residue_index=0)
        assert lb[AA_TO_ID["W"]].item() == 3.0
        # Forbidden mask should mark each omitted AA plus X
        fm = resolved.forbidden_mask(residue_index=0)
        for aa in "CDFG":
            assert bool(fm[AA_TO_ID[aa]])
        assert bool(fm[20])  # X always forbidden
