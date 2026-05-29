"""Precompute design results for the bundled UI examples.

Runs inference once per ``src/serving/examples/*.pdb`` and writes a sibling
``<id>.result.json`` matching the InferenceResult schema. The Gradio UI prefers
these precomputed results so larger ligand examples display instantly even
though live CPU inference on them would be slow.

Run this once on the deploy target (HF Spaces or the cluster), where torch and
the checkpoint are available::

    uv run python scripts/precompute_examples.py
    # or, with a higher cap so big examples are allowed:
    UMA_MAX_RESIDUES=1000 uv run python scripts/precompute_examples.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.serving.inference import InferenceEngine  # noqa: E402

EXAMPLES_DIR = Path(PROJECT_ROOT) / "src" / "serving" / "examples"


def main() -> None:
    pdbs = sorted(EXAMPLES_DIR.glob("*.pdb"))
    if not pdbs:
        print(f"no examples found in {EXAMPLES_DIR}")
        return

    # Allow larger structures than the live cap so all examples can be precomputed.
    cap = int(os.environ.get("UMA_MAX_RESIDUES", "1000"))
    engine = InferenceEngine(max_residues=cap)

    for pdb in pdbs:
        out = pdb.with_suffix(".result.json")
        print(f"precomputing {pdb.name} → {out.name} ...", flush=True)
        try:
            result = engine.run(
                pdb.read_text(encoding="utf-8"), n_samples=1, temperature=0.1, seed=0
            )
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"  SKIPPED ({type(exc).__name__}: {exc})")
            continue
        out.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
        print(
            f"  done: {result.n_residues} residues, "
            f"mean_confidence={result.mean_confidence:.3f}, {result.inference_ms:.0f} ms"
        )


if __name__ == "__main__":
    main()
