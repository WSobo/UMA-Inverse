"""Tests for the inference-facing extensions to the data layer.

Focus is on guarantees the inference path depends on:

* ``parse_pdb`` must return per-residue metadata that aligns 1:1 with the
  coordinate/sequence tensors.
* ``load_example_from_pdb(return_residue_ids=True)`` must thread IDs through
  the ``mask`` filter and the residue crop, preserving alignment.
* The legacy training-path call (no ``return_residue_ids``) must behave
  exactly as before — no extra keys leak into the returned dict.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.ligandmpnn_bridge import load_example_from_pdb
from src.data.pdb_parser import parse_pdb

FIXTURE = Path(__file__).parent.parent / "fixtures" / "1bc8.pdb"


@pytest.mark.skipif(not FIXTURE.exists(), reason="test fixture missing")
class TestParsePdbMetadata:
    def test_metadata_lengths_match_tensors(self) -> None:
        parsed = parse_pdb(str(FIXTURE))
        L = parsed["X"].shape[0]
        assert len(parsed["chain_ids"]) == L
        assert len(parsed["res_nums"]) == L
        assert len(parsed["insertion_codes"]) == L

    def test_metadata_types(self) -> None:
        parsed = parse_pdb(str(FIXTURE))
        assert all(isinstance(c, str) and len(c) == 1 for c in parsed["chain_ids"])
        assert all(isinstance(r, int) for r in parsed["res_nums"])
        assert all(isinstance(ic, str) for ic in parsed["insertion_codes"])

    def test_parse_chains_filter(self) -> None:
        parsed_all = parse_pdb(str(FIXTURE))
        all_chains = set(parsed_all["chain_ids"])
        if len(all_chains) < 2:
            pytest.skip("fixture has a single chain; filter test needs >= 2")
        one_chain = sorted(all_chains)[0]
        parsed_filtered = parse_pdb(str(FIXTURE), parse_chains=[one_chain])
        assert set(parsed_filtered["chain_ids"]) == {one_chain}


@pytest.mark.skipif(not FIXTURE.exists(), reason="test fixture missing")
class TestLoadExampleBackCompat:
    def test_legacy_call_omits_residue_ids(self) -> None:
        # Matches the datamodule + preprocess call signature exactly.
        example = load_example_from_pdb(
            pdb_path=str(FIXTURE),
            ligand_context_atoms=25,
            cutoff_for_score=8.0,
            max_total_nodes=384,
            device="cpu",
        )
        assert "residue_ids" not in example

    def test_legacy_tensor_keys_unchanged(self) -> None:
        example = load_example_from_pdb(str(FIXTURE))
        expected = {
            "residue_coords",
            "residue_features",
            "residue_mask",
            "sequence",
            "design_mask",
            "ligand_coords",
            "ligand_features",
            "ligand_mask",
        }
        assert set(example.keys()) == expected


@pytest.mark.skipif(not FIXTURE.exists(), reason="test fixture missing")
class TestLoadExampleWithResidueIds:
    def test_residue_ids_align_with_coords(self) -> None:
        example = load_example_from_pdb(
            str(FIXTURE), return_residue_ids=True, max_total_nodes=5000
        )
        L = example["residue_coords"].shape[0]
        assert len(example["residue_ids"]) == L

    def test_residue_id_format(self) -> None:
        example = load_example_from_pdb(
            str(FIXTURE), return_residue_ids=True, max_total_nodes=5000
        )
        # Every residue id starts with a single chain letter followed by digits
        for rid in example["residue_ids"][:10]:
            assert rid[0].isalpha()
            # The rest of the string must be parseable as an int + optional insertion
            rest = rid[1:]
            # Strip a trailing insertion-code letter if present
            if rest and rest[-1].isalpha():
                rest = rest[:-1]
            int(rest)  # must not raise

    def test_crop_preserves_id_alignment(self) -> None:
        # Force the crop path by requesting a tiny budget
        example = load_example_from_pdb(
            str(FIXTURE),
            return_residue_ids=True,
            max_total_nodes=40,  # ligand_context_atoms=25 by default, ~15 residues remain
        )
        L = example["residue_coords"].shape[0]
        assert len(example["residue_ids"]) == L
        # IDs should still all be well-formed after cropping
        for rid in example["residue_ids"]:
            assert rid[0].isalpha()
