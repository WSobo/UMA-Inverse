# Deploying the serving layer to Hugging Face Spaces (Docker SDK)

The Space runs the CPU serving image (`Dockerfile`) and serves the Gradio UI +
REST API on port 7860. The model weights are pulled from the HF Hub at startup,
not baked into the image.

## Prerequisites

- An HF account (the weights live under `WSobo/Uma-Inverse-1`).
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
| `UMA_MAX_RESIDUES` | 120 | Live-endpoint residue cap (413 above it). |
| `UMA_MAX_CONCURRENCY` | 1 | Serialise inference on the 2-vCPU box. |
| `UMA_REQUEST_TIMEOUT_S` | 300 | Client-facing timeout backstop. |
| `UMA_TORCH_THREADS` | 2 | Pin CPU threads to the Space's 2 vCPUs. |
| `UMA_CKPT_REPO` / `UMA_CKPT_FILE` | `WSobo/Uma-Inverse-1` / canonical | Override the weights source. |

## 5. Agent (MCP) usage

Point the MCP server at the deployed Space and run it for an agent client:

```bash
UMA_API_URL=https://<user>-uma-inverse.hf.space uv run python -m src.mcp.server
```

The tool `design_sequence_for_structure(pdb, ligand?, temperature?)` calls
`/design` and returns a markdown result.
