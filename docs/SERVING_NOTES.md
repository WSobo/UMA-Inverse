# Serving notes — inference path reconnaissance

Reference for the serving layer: how the existing inference code is reused, and
the facts that shaped the design. Nothing here retrains or reimplements the
model — the serving layer wraps the same path the CLI (`uma-inverse design`) uses.

## Package layout

The installed package is **flat `src/`** (`[tool.hatch.build.targets.wheel] packages = ["src"]`).
Serving code lives in `src/serving/` and imports `from src.inference... import ...`.

## Inference API reused

| Concern | Symbol | File |
|---|---|---|
| Checkpoint resolve / HF fetch | `resolve_checkpoint`, `fetch_default_checkpoint` | `src/inference/weights.py` |
| Load model once, encode a PDB | `InferenceSession.from_checkpoint`, `.load_structure` | `src/inference/session.py` |
| Sample sequences | `autoregressive_design` → `list[DesignSample]` | `src/inference/decoding.py` |
| Default "all designable" constraints | `DesignConstraints.from_cli().resolve(ctx)` | `src/inference/constraints.py` |
| Token→AA, confidence | `ids_to_sequence`, `DesignSample.overall_confidence` | `src/utils/io.py`, `decoding.py` |

Key facts:

- **`load_structure` takes a file path, not a string.** The serving adapter writes
  the posted PDB to a temp file (`src/serving/inference.py`).
- The model loads on CPU mechanically: `_load_weights` already uses
  `torch.load(map_location="cpu")`; `_resolve_device("cpu")` works; no hardcoded
  `.cuda()`. Only **speed** is the question (see below).
- `forward()` returns `logits [B, L, 21]` (20 AA + X). Per-residue confidence is
  the softmax max-probability per position; the aggregate is the existing
  LigandMPNN-style `overall_confidence = exp(mean log p of sampled residues)`.
- `configs/config.yaml` is self-contained (`model:`/`data:` inline with OmegaConf
  interpolation), which is exactly what `from_checkpoint` loads — so serving
  points at it directly.
- The checkpoint `WSobo/Uma-Inverse-1` / `uma-inverse-23-1.2093.ckpt` is fetched
  from the HF Hub on first use; never committed.

## Decoding choice

Serving uses **autoregressive decoding** (matching `scripts/SLURM/04_inference.sh`):
the decoder runs once per residue, so latency scales with structure length. A
faster Gibbs path (`gibbs_design`) exists but was not chosen for the served
default.

## CPU latency reality (the central constraint)

Autoregressive decoding on a 2-vCPU CPU box (HF Spaces CPU Basic) is **slow** and
length-dependent — a ~140-residue structure is plausibly tens of seconds, larger
ones minutes. Mitigations, all honest:

1. **Residue cap** (`UMA_MAX_RESIDUES`, default 120): oversized structures get
   `413` *before* any decode. This is the real protection, not the timeout.
2. **Precomputed examples**: `scripts/precompute_examples.py` writes
   `<id>.result.json` next to each bundled example; the UI serves those instantly.
3. **Honest framing**: README/Space say "live inference works for small proteins
   on free CPU; production would use a GPU endpoint." No faked latency.

> Real measured CPU latency is captured during HF Spaces testing and recorded in
> the README's Serving section. (The local dev checkout has no GPU/torch env, so
> timing is done on the deploy target.)

## Concurrency

The timeout (`UMA_REQUEST_TIMEOUT_S`) is a **response-level backstop only** — a
running Python inference thread can't be cancelled, so an orphaned decode runs to
completion. A semaphore (`UMA_MAX_CONCURRENCY`, default 1) serialises inference so
concurrent requests queue instead of thrashing the cores, which keeps the latency
percentiles meaningful.
