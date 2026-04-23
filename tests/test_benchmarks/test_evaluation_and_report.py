"""Integration tests for :mod:`src.benchmarks.evaluation` and
:mod:`src.benchmarks.report`.

End-to-end: load a tiny fixture PDB, run teacher-forced evaluation with a
random-init model, assemble the report tables, verify file schemas.
These tests take ~30s total on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.benchmarks.evaluation import evaluate_single_pdb, evaluate_validation_set
from src.benchmarks.report import per_pdb_frame, per_position_frame, write_report
from src.inference.session import InferenceSession

FIXTURE_PDB = Path(__file__).parent.parent / "fixtures" / "1bc8.pdb"
FIXTURE_4GYT = Path(__file__).parent.parent / "fixtures" / "4gyt.pdb"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession.from_checkpoint(
        config_path=CONFIG_PATH, checkpoint=None, device="cpu"
    )


@pytest.mark.skipif(not FIXTURE_PDB.exists(), reason="fixture missing")
class TestEvaluateSinglePdb:
    def test_returns_result(self, session: InferenceSession) -> None:
        result = evaluate_single_pdb(session, FIXTURE_PDB, max_total_nodes=5000)
        assert result is not None
        assert result.pdb_id == "1bc8"
        assert result.num_residues > 0
        assert 0.0 <= result.recovery <= 1.0
        assert result.probs_full.shape == (result.num_residues, 21)

    def test_per_position_count_matches_non_x_residues(
        self, session: InferenceSession
    ) -> None:
        result = evaluate_single_pdb(session, FIXTURE_PDB, max_total_nodes=5000)
        # Every record should have a non-X native token
        assert all(r.native_token != 20 for r in result.per_position)

    def test_mask_ligand_changes_probs(self, session: InferenceSession) -> None:
        with_ligand = evaluate_single_pdb(session, FIXTURE_PDB, max_total_nodes=5000)
        no_ligand = evaluate_single_pdb(
            session, FIXTURE_PDB, max_total_nodes=5000, mask_ligand=True
        )
        # Random init so values don't match a reference, but two different
        # inputs should give different distributions somewhere.
        assert with_ligand.probs_full.shape == no_ligand.probs_full.shape
        assert not (with_ligand.probs_full == no_ligand.probs_full).all()

    def test_missing_file_returns_none(
        self, session: InferenceSession, tmp_path: Path
    ) -> None:
        ghost = tmp_path / "ghost.pdb"
        ghost.write_text("not a real pdb")
        assert evaluate_single_pdb(session, ghost) is None


@pytest.mark.skipif(
    not (FIXTURE_PDB.exists() and FIXTURE_4GYT.exists()),
    reason="fixtures missing",
)
class TestEvaluateValidationSet:
    @pytest.fixture(scope="class")
    def tiny_val_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Set up a tiny val.json pointing to the fixtures directory."""
        d = tmp_path_factory.mktemp("valset")
        val_json = d / "val.json"
        val_json.write_text(json.dumps(["1bc8", "4gyt"]))
        return d

    def test_evaluates_both_fixtures(
        self, session: InferenceSession, tiny_val_dir: Path
    ) -> None:
        results = evaluate_validation_set(
            session=session,
            val_json=tiny_val_dir / "val.json",
            pdb_dir=FIXTURE_PDB.parent,
            max_total_nodes=5000,
        )
        assert len(results) == 2
        ids = {r.pdb_id for r in results}
        assert ids == {"1bc8", "4gyt"}

    def test_n_pdbs_subsample(
        self, session: InferenceSession, tiny_val_dir: Path
    ) -> None:
        results = evaluate_validation_set(
            session=session,
            val_json=tiny_val_dir / "val.json",
            pdb_dir=FIXTURE_PDB.parent,
            n_pdbs=1,
            seed=0,
            max_total_nodes=5000,
        )
        assert len(results) == 1

    def test_missing_pdb_is_skipped(
        self, session: InferenceSession, tiny_val_dir: Path, tmp_path: Path
    ) -> None:
        # Write a val.json with one real id and one that won't resolve
        val = tmp_path / "mixed.json"
        val.write_text(json.dumps(["1bc8", "nonexistent_id"]))
        results = evaluate_validation_set(
            session=session,
            val_json=val,
            pdb_dir=FIXTURE_PDB.parent,
            max_total_nodes=5000,
        )
        assert {r.pdb_id for r in results} == {"1bc8"}


@pytest.mark.skipif(not FIXTURE_PDB.exists(), reason="fixture missing")
class TestReportWriter:
    def test_full_pipeline(
        self, session: InferenceSession, tmp_path: Path
    ) -> None:
        result = evaluate_single_pdb(session, FIXTURE_PDB, max_total_nodes=5000)
        assert result is not None

        write_report(
            tmp_path,
            evaluations=[result],
            run_metadata={"checkpoint_path": "fake.ckpt", "git_hash": "abc123"},
        )

        # Every required artefact
        expected = [
            "summary.md",
            "summary.json",
            "per_pdb.csv",
            "per_position.parquet",
            "confusion_matrix.csv",
            "calibration.csv",
            "aa_composition.csv",
            "figures/confusion.png",
            "figures/calibration.png",
            "figures/near_ligand.png",
            "figures/aa_composition.png",
            "figures/perplexity_by_length.png",
        ]
        for rel in expected:
            assert (tmp_path / rel).exists(), f"missing artefact: {rel}"

    def test_summary_schema(
        self, session: InferenceSession, tmp_path: Path
    ) -> None:
        result = evaluate_single_pdb(session, FIXTURE_PDB, max_total_nodes=5000)
        write_report(tmp_path, evaluations=[result])
        payload = json.loads((tmp_path / "summary.json").read_text())
        head = payload["headline"]
        for key in (
            "num_pdbs",
            "overall_recovery",
            "perplexity",
            "expected_calibration_error",
            "per_aa_recovery",
            "native_aa_composition",
            "predicted_aa_composition",
        ):
            assert key in head

    def test_per_position_parquet_columns(
        self, session: InferenceSession, tmp_path: Path
    ) -> None:
        result = evaluate_single_pdb(session, FIXTURE_PDB, max_total_nodes=5000)
        write_report(tmp_path, evaluations=[result])
        df = pd.read_parquet(tmp_path / "per_position.parquet")
        expected_cols = {
            "pdb_id",
            "residue_id",
            "chain_id",
            "position",
            "native_token",
            "pred_token",
            "native_aa",
            "pred_aa",
            "native_log_prob",
            "entropy",
            "distance_to_ligand",
            "ligand_context_masked",
        }
        assert expected_cols.issubset(df.columns)
        assert len(df) == len(result.per_position)


class TestFrameBuilders:
    def test_per_pdb_frame_shape(self, session: InferenceSession) -> None:
        r = evaluate_single_pdb(session, FIXTURE_PDB, max_total_nodes=5000)
        df = per_pdb_frame([r])
        assert list(df.columns) == [
            "pdb_id", "num_residues", "recovery",
            "mean_log_prob", "perplexity", "wall_seconds",
        ]
        assert len(df) == 1

    def test_per_position_frame_shape(self, session: InferenceSession) -> None:
        r = evaluate_single_pdb(session, FIXTURE_PDB, max_total_nodes=5000)
        df = per_position_frame([r])
        assert len(df) == len(r.per_position)
        assert "native_aa" in df.columns
        assert "pred_aa" in df.columns
