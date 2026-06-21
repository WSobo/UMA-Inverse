# Deploying the serving layer to Hugging Face Spaces (Docker SDK)

The Space runs the CPU serving image (`Dockerfile`) and serves the Gradio UI +
REST API on port 7860. The model weights are pulled from the HF Hub at startup,
not baked into the image.

> **Checkpoint compatibility.** The canonical HF weights — `WSobo/UMA-Inverse`,
> `uma-inverse-11-1.2175.ckpt` (the v5 model, epoch 11) — match this repo's code
> and are **auto-fetched at startup**, so no configuration is needed locally or
> on the Space. `configs/config.yaml` is the v5 architecture, and a served
> checkpoint must match it. To serve a different checkpoint, set `UMA_CKPT_REPO`
> / `UMA_CKPT_FILE` (an HF source, §4) or `UMA_CKPT` (a local file path). The
> older `WSobo/Uma-Inverse-1` (v1) weights do **not** match the v5 code and will
> load only partially.

## Prerequisites

- An HF account (the weights live under `WSobo/UMA-Inverse`).
- An HF token with **write** scope: <https://huggingface.co/settings/tokens>.
- `huggingface_hub` CLI (`uv run hf auth login`, or `pip install -U huggingface_hub`).

## 1. Local verification (do this first)

```bash
docker build -t uma-inverse-serving .

# Confirm the image is CPU-only — should print a +cpu torch and no nvidia-* pkgs.
docker run --rm uma-inverse-serving python -c "import torch; print(torch.__version__)"

# Run it (downloads the checkpoint from the Hub on first request/startup).
docker run --rm -p 7860:7860 uma-inverse-serving
```

Then, against `http://localhost:7860`:

```bash
curl -s localhost:7860/health
PDB=$(cat src/serving/examples/1crn.pdb)
curl -s -X POST localhost:7860/design \
  -H 'Content-Type: application/json' \
  --argjson ... # or use a small Python/jq helper to embed the PDB safely
curl -s localhost:7860/metrics | grep uma_   # counters should appear/increment
open http://localhost:7860/        # Gradio UI
open http://localhost:7860/docs    # OpenAPI
```

Record the observed CPU latency (the `X-Inference-MS` header / the UI's latency
readout) and put the real number in the README Serving section.

## 2. (Optional) Precompute example results for a snappy UI

On the deploy target (where torch + the checkpoint are available):

```bash
UMA_MAX_RESIDUES=1000 uv run python scripts/precompute_examples.py
```

This writes `src/serving/examples/<id>.result.json`, which the UI serves
instantly. Commit those JSON files so the Space ships with them.

## 3. Create the Space and push

```bash
# Create a Docker Space (public) under your account.
hf repo create uma-inverse --repo-type space --space_sdk docker

# Point the Space's README at the prepared front-matter, then push the repo.
cp deploy/hf_space/README.md README_SPACE.md   # keep the GitHub README separate
git remote add space https://huggingface.co/spaces/<user>/uma-inverse
# The Space needs deploy/hf_space/README.md as its top-level README.md:
#   either commit it as README.md on a deploy branch, or push a subtree.
git push space HEAD:main
```

The Space build uses the root `Dockerfile`. After the build, the first request
triggers the checkpoint download (cold start); the startup warm-up primes the
model so subsequent requests skip cold-load cost.

## 4. Configuration (Space → Settings → Variables)

| Variable | Default | Purpose |
|---|---|---|
| `UMA_MAX_RESIDUES` | 600 | Residue cap (413 above it) — a memory guard (dense O(N²) model), not a speed limit. |
| `UMA_MAX_CONCURRENCY` | 1 | Serialise inference on the 2-vCPU box. |
| `UMA_REQUEST_TIMEOUT_S` | 300 | Client-facing timeout backstop. |
| `UMA_TORCH_THREADS` | 2 | Pin CPU threads to the Space's 2 vCPUs. |
| `UMA_CKPT_REPO` / `UMA_CKPT_FILE` | `WSobo/UMA-Inverse` / `uma-inverse-11-1.2175.ckpt` | Override the HF weights source. |

## 5. Agent (MCP) usage

Point the MCP server at the deployed Space and run it for an agent client:

```bash
UMA_API_URL=https://<user>-uma-inverse.hf.space uv run python -m src.mcp.server
```

The tool `design_sequence_for_structure(pdb, ligand?, temperature?)` calls
`/design` and returns a markdown result.
