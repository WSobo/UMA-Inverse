"""Tests for :mod:`src.inference.batch` and :mod:`src.inference.output`.

Cover the serialisation / resume / override logic so long-running screens
recover predictably after interruptions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.inference.batch import (
    append_done,
    filter_pending,
    load_batch_spec,
    load_done_set,
    merge_constraint_kwargs,
)
from src.inference.decoding import DesignSample
from src.inference.output import (
    build_manifest,
    build_ranked_rows,
    write_per_residue_confidence,
    write_probs_npz,
    write_ranked_csv,
    write_samples_fasta,
)
from src.inference.session import StructureContext

# ─── Batch spec ───────────────────────────────────────────────────────────────


class TestBatchSpec:
    def test_empty_overrides(self, tmp_path: Path) -> None:
        spec = {str(tmp_path / "a.pdb"): {}, str(tmp_path / "b.pdb"): None}
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        # Touch files so the spec resolves absolutely
        (tmp_path / "a.pdb").touch()
        (tmp_path / "b.pdb").touch()
        entries = load_batch_spec(path)
        assert len(entries) == 2
        assert all(e.overrides == {} for e in entries)

    def test_relative_paths_resolved(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.pdb").touch()
        path = sub / "spec.json"
        path.write_text(json.dumps({"c.pdb": {}}))
        entries = load_batch_spec(path)
        assert entries[0].pdb_path == (sub / "c.pdb").resolve()

    def test_override_keys_validated(self, tmp_path: Path) -> None:
        spec = {"p.pdb": {"bogus_key": 1}}
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match="unknown override keys"):
            load_batch_spec(path)

    def test_non_dict_override_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps({"p.pdb": "invalid-string"}))
        with pytest.raises(ValueError, match="must be a mapping"):
            load_batch_spec(path)


class TestResumeSupport:
    def test_empty_done_file(self, tmp_path: Path) -> None:
        assert load_done_set(tmp_path / "missing.txt") == set()

    def test_append_and_read(self, tmp_path: Path) -> None:
        done = tmp_path / "done.txt"
        append_done(done, "/p/a.pdb")
        append_done(done, "/p/b.pdb")
        assert load_done_set(done) == {"/p/a.pdb", "/p/b.pdb"}

    def test_filter_pending_skips_done(self, tmp_path: Path) -> None:
        from src.inference.batch import BatchEntry

        done = tmp_path / "done.txt"
        append_done(done, str(tmp_path / "a.pdb"))
        entries = [
            BatchEntry(pdb_path=tmp_path / "a.pdb"),
            BatchEntry(pdb_path=tmp_path / "b.pdb"),
        ]
        remaining = filter_pending(entries, done_path=done, resume=True)
        assert len(remaining) == 1
        assert remaining[0].pdb_path == tmp_path / "b.pdb"

    def test_filter_pending_no_resume_returns_all(self, tmp_path: Path) -> None:
        from src.inference.batch import BatchEntry

        entries = [BatchEntry(pdb_path=tmp_path / "a.pdb")]
        assert len(filter_pending(entries, done_path=tmp_path / "done.txt", resume=False)) == 1


class TestMergeConstraintKwargs:
    def test_overrides_replace_defaults(self) -> None:
        defaults = {"fix": "A1", "bias": "W:1.0"}
        overrides = {"fix": "B5"}
        merged = merge_constraint_kwargs(defaults, overrides)
        assert merged["fix"] == "B5"
        assert merged["bias"] == "W:1.0"

    def test_override_introduces_new_key(self) -> None:
        merged = merge_constraint_kwargs({}, {"bias": "W:2.0"})
        assert merged == {"bias": "W:2.0"}


# ─── Manifest ─────────────────────────────────────────────────────────────────


class TestManifest:
    def test_writable_and_readable(self, tmp_path: Path) -> None:
        m = build_manifest(
            run_name="unit-test",
            command="uma-inverse design --pdb x.pdb",
            checkpoint_path=None,
            config_path=None,
            config_snapshot={"model": {"node_dim": 128}},
            seed=42,
            temperature=0.1,
            top_p=0.95,
            decoding_order="random",
            num_pdbs=1,
            num_samples_per_pdb=5,
        )
        out = tmp_path / "manifest.json"
        m.write(out)
        loaded = json.loads(out.read_text())
        assert loaded["run_name"] == "unit-test"
        assert loaded["seed"] == 42
        assert loaded["manifest_version"] == "1"
        assert loaded["hostname"]  # non-empty
        assert "start_timestamp" in loaded


# ─── Output writers ───────────────────────────────────────────────────────────


def _fake_context(L: int = 6, pdb_id: str = "FAKE", device: torch.device = torch.device("cpu")) -> StructureContext:
    residue_ids = [f"A{i+1}" for i in range(L)]
    return StructureContext(
        residue_ids=residue_ids,
        residue_id_to_index={rid: i for i, rid in enumerate(residue_ids)},
        chain_ids=["A"] * L,
        native_sequence=torch.tensor([0, 1, 2, 3, 4, 5][:L], dtype=torch.long),
        design_mask=torch.ones(L, dtype=torch.bool),
        z=torch.zeros(1, L, L, 4),
        node_repr_res=torch.zeros(1, L, 4),
        node_repr=torch.zeros(1, L, 4),
        lig_ctx=torch.zeros(1, L, 4),
        pair_mask=torch.ones(1, L, L, dtype=torch.bool),
        residue_mask=torch.ones(1, L, dtype=torch.bool),
        residue_count=L,
        pdb_path=f"/fake/{pdb_id}.pdb",
        device=device,
    )


def _fake_sample(L: int = 6, seed: int = 42) -> DesignSample:
    probs = torch.zeros(L, 21)
    token_ids = torch.randint(0, 20, (L,))
    for i, t in enumerate(token_ids.tolist()):
        probs[i, int(t)] = 0.9
        probs[i, (int(t) + 1) % 20] = 0.1
    log_probs = torch.log(probs.gather(1, token_ids.unsqueeze(-1)).squeeze(-1))
    return DesignSample(
        token_ids=token_ids,
        log_probs=log_probs,
        probs_full=probs,
        decoding_order=torch.arange(L),
        seed=seed,
        temperature=0.1,
        top_p=0.95,
    )


class TestFastaWriter:
    def test_writes_native_plus_samples(self, tmp_path: Path) -> None:
        ctx = _fake_context(L=6)
        samples = [_fake_sample(L=6, seed=i) for i in range(3)]
        path = tmp_path / "out.fa"
        write_samples_fasta(
            path,
            pdb_id="FAKE",
            ctx=ctx,
            samples=samples,
            designable_mask=torch.ones(6, dtype=torch.bool),
        )
        content = path.read_text().splitlines()
        # 1 native + 3 samples = 4 records = 8 lines
        assert len(content) == 8
        headers = [line for line in content if line.startswith(">")]
        assert "kind=native" in headers[0]
        assert "sample=0" in headers[1]
        assert "seed=0" in headers[1]


class TestConfidenceJson:
    def test_schema(self, tmp_path: Path) -> None:
        ctx = _fake_context(L=4)
        samples = [_fake_sample(L=4, seed=7)]
        out = tmp_path / "conf.json"
        write_per_residue_confidence(
            out, pdb_id="FAKE", ctx=ctx, samples=samples,
            designable_mask=torch.ones(4, dtype=torch.bool),
        )
        data = json.loads(out.read_text())
        assert data["num_residues"] == 4
        assert len(data["residue_ids"]) == 4
        assert len(data["samples"]) == 1
        pos = data["samples"][0]["positions"][0]
        assert "top_k" in pos and len(pos["top_k"]) > 0
        assert "entropy" in pos
        assert 0.0 <= pos["sampled_prob"] <= 1.0


class TestProbsNpz:
    def test_roundtrip(self, tmp_path: Path) -> None:
        import numpy as np

        ctx = _fake_context(L=5)
        samples = [_fake_sample(L=5, seed=i) for i in range(2)]
        out = tmp_path / "probs.npz"
        write_probs_npz(out, pdb_id="FAKE", ctx=ctx, samples=samples)
        loaded = np.load(out, allow_pickle=True)
        assert loaded["probs"].shape == (2, 5, 21)
        assert loaded["token_ids"].shape == (2, 5)
        assert loaded["seeds"].tolist() == [0, 1]
        assert str(loaded["pdb_id"]) == "FAKE"


class TestRankedCsv:
    def test_deduplication_and_ranking(self, tmp_path: Path) -> None:
        ctx = _fake_context(L=4)
        # Produce two identical samples (should dedup) + one distinct
        s1 = _fake_sample(L=4, seed=1)
        s2 = DesignSample(
            token_ids=s1.token_ids.clone(),
            log_probs=s1.log_probs.clone(),
            probs_full=s1.probs_full.clone(),
            decoding_order=s1.decoding_order.clone(),
            seed=2,
            temperature=0.1,
        )
        s3 = _fake_sample(L=4, seed=3)
        rows = build_ranked_rows(
            pdb_id="FAKE", ctx=ctx, samples=[s1, s2, s3],
            designable_mask=torch.ones(4, dtype=torch.bool),
        )
        # At most 2 distinct sequences
        assert len(rows) <= 2
        # Seeds of merged samples are concatenated
        merged = [r for r in rows if len(r.sample_seeds) > 1]
        assert len(merged) == 1
        assert set(merged[0].sample_seeds) == {1, 2}
        # Rows sorted by ascending NLL
        nlls = [r.mean_nll for r in rows]
        assert nlls == sorted(nlls)

    def test_csv_appendable(self, tmp_path: Path) -> None:
        ctx = _fake_context(L=3)
        rows = build_ranked_rows(
            pdb_id="P", ctx=ctx, samples=[_fake_sample(L=3, seed=1)],
            designable_mask=torch.ones(3, dtype=torch.bool),
        )
        path = tmp_path / "ranked.csv"
        write_ranked_csv(path, rows)
        write_ranked_csv(path, rows)  # second call should append (no duplicate header)
        lines = path.read_text().splitlines()
        assert sum(1 for ln in lines if ln.startswith("pdb_id,")) == 1
