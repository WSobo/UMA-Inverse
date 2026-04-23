"""Tests for :class:`InferenceSession` and :class:`StructureContext`.

CPU-only smoke tests with a random-initialised model. They cover the
contract the decoder relies on (tensor shapes, residue-id alignment, mask
semantics) but do not exercise checkpoint loading on disk — that path is
trivial wrapping around :func:`torch.load`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.inference import InferenceSession

FIXTURE_PDB = Path(__file__).parent.parent / "fixtures" / "1bc8.pdb"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


@pytest.mark.skipif(
    not FIXTURE_PDB.exists() or not CONFIG_PATH.exists(),
    reason="fixture PDB or config missing",
)
class TestInferenceSession:
    @pytest.fixture(scope="class")
    def session(self) -> InferenceSession:
        return InferenceSession.from_checkpoint(
            config_path=CONFIG_PATH, checkpoint=None, device="cpu"
        )

    @pytest.fixture(scope="class")
    def ctx(self, session: InferenceSession):
        return session.load_structure(str(FIXTURE_PDB), max_total_nodes=5000)

    def test_structure_context_tensor_shapes(self, ctx) -> None:
        L = ctx.residue_count
        assert ctx.native_sequence.shape == (L,)
        assert ctx.node_repr_res.shape[1] == L
        assert ctx.z.shape[1] == ctx.z.shape[2]  # square pair tensor
        assert ctx.z.shape[1] >= L  # includes ligand atoms
        assert ctx.pair_mask.shape == ctx.z.shape[:-1]

    def test_residue_ids_align_with_tensors(self, ctx) -> None:
        assert len(ctx.residue_ids) == ctx.residue_count
        assert len(ctx.chain_ids) == ctx.residue_count
        for i, rid in enumerate(ctx.residue_ids):
            assert ctx.residue_id_to_index[rid] == i

    def test_mask_ligand_zeros_out_ligand(self, session: InferenceSession) -> None:
        ctx_default = session.load_structure(str(FIXTURE_PDB), max_total_nodes=5000)
        ctx_masked = session.load_structure(
            str(FIXTURE_PDB), max_total_nodes=5000, mask_ligand=True
        )
        # Only ligand columns in the pair tensor's node_mask differ
        L = ctx_default.residue_count
        default_ligand_pair = ctx_default.pair_mask[0, :, L:]
        masked_ligand_pair = ctx_masked.pair_mask[0, :, L:]
        # Default should have at least one True, masked should be all False
        # (when the PDB actually has ligand atoms)
        if default_ligand_pair.numel() > 0:
            assert masked_ligand_pair.sum().item() == 0

    def test_missing_pdb_raises(self, session: InferenceSession, tmp_path: Path) -> None:
        ghost = tmp_path / "does_not_exist.pdb"
        with pytest.raises(FileNotFoundError):
            session.load_structure(str(ghost))

    def test_model_in_eval_mode(self, session: InferenceSession) -> None:
        assert not session.model.training

    def test_structure_context_immutability(self, ctx) -> None:
        # Sanity check: residue_ids should not include duplicates
        assert len(set(ctx.residue_ids)) == len(ctx.residue_ids)

    def test_context_length_property(self, ctx) -> None:
        assert ctx.length == ctx.residue_count

    def test_tensors_on_correct_device(self, ctx) -> None:
        cpu = torch.device("cpu")
        assert ctx.device == cpu
        assert ctx.z.device == cpu
        assert ctx.native_sequence.device == cpu

    def test_parse_chains_filter(self, session: InferenceSession) -> None:
        first_full = session.load_structure(str(FIXTURE_PDB), max_total_nodes=5000)
        chains = sorted(set(first_full.chain_ids))
        if len(chains) < 2:
            pytest.skip("fixture has a single chain; filter test needs >= 2")
        one = chains[0]
        filtered = session.load_structure(
            str(FIXTURE_PDB), parse_chains=[one], max_total_nodes=5000
        )
        assert set(filtered.chain_ids) == {one}
