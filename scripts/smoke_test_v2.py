"""CPU-only smoke test for UMA-Inverse v2 featurizer phases.

Each v2 phase is config-gated and must leave the v1 default path untouched.
This script exercises both paths on a small fixture PDB (no network, no GPU,
no writes to data/, checkpoints/, outputs/, or logs/) and asserts the
invariants that would break if a phase silently regressed.

The smoke test is committed alongside Phase 1 so each phase commit can be
independently verified. It grows with the phases — Phase 2 adds a Cβ-anchor
test, Phase 3 adds a multi-backbone-distance test.

Run:
    uv run python scripts/smoke_test_v2.py
"""
from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.ligandmpnn_bridge import load_example_from_pdb
from src.models.uma_inverse import UMAInverse

FIXTURE_PDB = PROJECT_ROOT / "tests" / "fixtures" / "1bc8.pdb"


def _small_config(ligand_featurizer: str = "onehot6") -> dict:
    """Return a tiny UMAInverse config that runs quickly on CPU.

    Shapes are reduced across the board; the smoke test only needs to confirm
    forward-pass plumbing, not learn anything.
    """
    return {
        "residue_input_dim": 6,
        "ligand_input_dim": 6,
        "node_dim": 32,
        "pair_dim": 32,
        "pair_hidden_dim": 32,
        "num_pairmixer_blocks": 2,
        "pair_transition_mult": 2,
        "num_rbf": 16,
        "max_distance": 24.0,
        "dropout": 0.0,
        "gradient_checkpointing": False,
        "thermal_noise_std": 0.0,
        "relpos_max": 16,
        "ar_num_heads": 4,
        "ligand_featurizer": ligand_featurizer,
    }


def _build_batch(example: dict) -> dict:
    """Unsqueeze a single-structure example dict into a batch of size 1."""
    batch = {}
    for k, v in example.items():
        if isinstance(v, torch.Tensor):
            if k == "sequence":
                batch[k] = v.long().unsqueeze(0)
            elif v.dtype == torch.bool:
                batch[k] = v.unsqueeze(0)
            elif v.dtype == torch.long:
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = v.float().unsqueeze(0)
        else:
            batch[k] = v
    return batch


# ── Tests ────────────────────────────────────────────────────────────────────


def test_v1_path() -> None:
    example = load_example_from_pdb(
        str(FIXTURE_PDB), max_total_nodes=256, ligand_featurizer="onehot6",
    )
    assert "ligand_features" in example, "v1 path must emit ligand_features"
    assert "ligand_atomic_numbers" not in example, \
        "v1 path must not emit ligand_atomic_numbers"
    assert example["ligand_features"].dtype == torch.float32, \
        f"ligand_features should be float, got {example['ligand_features'].dtype}"
    assert example["ligand_features"].dim() == 2 and \
           example["ligand_features"].shape[1] == 6, \
        f"ligand_features shape should be [M, 6], got {tuple(example['ligand_features'].shape)}"

    batch = _build_batch(example)
    L = batch["residue_coords"].shape[1]
    M = batch["ligand_coords"].shape[1]
    assert batch["ligand_features"].shape == (1, M, 6), \
        f"batched ligand_features should be [1, M, 6], got {tuple(batch['ligand_features'].shape)}"

    model = UMAInverse(_small_config("onehot6")).eval()
    with torch.no_grad():
        out = model(batch)
    logits = out["logits"]
    assert logits.shape == (1, L, 21), \
        f"expected logits shape (1, {L}, 21), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "v1 logits contain NaN or Inf"


def test_v2_phase1() -> None:
    example = load_example_from_pdb(
        str(FIXTURE_PDB), max_total_nodes=256,
        ligand_featurizer="atomic_number_embedding",
    )
    assert "ligand_atomic_numbers" in example, \
        "embedding path must emit ligand_atomic_numbers"
    assert "ligand_features" not in example, \
        "embedding path must not emit ligand_features"
    z = example["ligand_atomic_numbers"]
    assert z.dtype == torch.long, \
        f"ligand_atomic_numbers should be long, got {z.dtype}"
    assert z.dim() == 1, f"ligand_atomic_numbers should be [M], got shape {tuple(z.shape)}"
    if z.numel() > 0:
        assert int(z.min()) >= 0, f"ligand_atomic_numbers min={int(z.min())} < 0"
        assert int(z.max()) <= 119, \
            f"ligand_atomic_numbers max={int(z.max())} > 119 (unknown-element slot)"

    batch = _build_batch(example)
    L = batch["residue_coords"].shape[1]
    M = batch["ligand_coords"].shape[1]
    assert batch["ligand_atomic_numbers"].shape == (1, M), \
        f"batched ligand_atomic_numbers should be [1, M], got {tuple(batch['ligand_atomic_numbers'].shape)}"

    model = UMAInverse(_small_config("atomic_number_embedding")).eval()
    assert isinstance(model.ligand_in, torch.nn.Embedding), \
        f"expected nn.Embedding for ligand_in, got {type(model.ligand_in).__name__}"
    assert model.ligand_in.num_embeddings == 120, \
        f"expected vocab 120, got {model.ligand_in.num_embeddings}"
    assert model.ligand_in.padding_idx == 0, \
        f"expected padding_idx 0, got {model.ligand_in.padding_idx}"
    with torch.no_grad():
        out = model(batch)
    logits = out["logits"]
    assert logits.shape == (1, L, 21), \
        f"expected logits shape (1, {L}, 21), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "v2 phase-1 logits contain NaN or Inf"


def test_strict_load_mismatch_v1_to_v2() -> None:
    """v1 checkpoint state_dict must NOT load into a v2 model under strict=True."""
    v1_model = UMAInverse(_small_config("onehot6"))
    v2_model = UMAInverse(_small_config("atomic_number_embedding"))
    try:
        v2_model.load_state_dict(v1_model.state_dict(), strict=True)
    except RuntimeError:
        return  # expected: keys / shapes don't match
    raise AssertionError(
        "v2 model silently accepted a v1 state_dict under strict=True — "
        "ligand_in changed type (Linear→Embedding) and the shapes must not align"
    )


def test_strict_load_mismatch_v2_to_v1() -> None:
    """v2 checkpoint state_dict must NOT load into a v1 model under strict=True."""
    v1_model = UMAInverse(_small_config("onehot6"))
    v2_model = UMAInverse(_small_config("atomic_number_embedding"))
    try:
        v1_model.load_state_dict(v2_model.state_dict(), strict=True)
    except RuntimeError:
        return  # expected: keys / shapes don't match
    raise AssertionError(
        "v1 model silently accepted a v2 state_dict under strict=True — "
        "ligand_in changed type (Embedding→Linear) and the shapes must not align"
    )


# ── Runner ───────────────────────────────────────────────────────────────────


@dataclass
class _Result:
    name: str
    passed: bool
    error: str = ""


def _run(name: str, fn: Callable[[], None]) -> _Result:
    try:
        fn()
        return _Result(name, True)
    except Exception as exc:
        return _Result(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def main() -> int:
    if not FIXTURE_PDB.exists():
        print(f"FAIL: fixture not found: {FIXTURE_PDB}")
        return 1

    tests: list[tuple[str, Callable[[], None]]] = [
        ("test_v1_path",                         test_v1_path),
        ("test_v2_phase1",                       test_v2_phase1),
        ("test_strict_load_mismatch_v1_to_v2",   test_strict_load_mismatch_v1_to_v2),
        ("test_strict_load_mismatch_v2_to_v1",   test_strict_load_mismatch_v2_to_v1),
    ]

    results = [_run(name, fn) for name, fn in tests]

    print()
    print("─" * 70)
    print(f"{'test':50s}  {'result':>8s}")
    print("─" * 70)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:50s}  {status:>8s}")
    print("─" * 70)

    failed = [r for r in results if not r.passed]
    if failed:
        print()
        for r in failed:
            print(f"── {r.name} ──")
            print(r.error)
        print(f"\n{len(failed)}/{len(results)} tests failed.")
        return 1

    print(f"\n{len(results)}/{len(results)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
