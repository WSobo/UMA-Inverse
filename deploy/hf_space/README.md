---
title: UMA-Inverse
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Ligand-conditioned protein inverse folding, served on CPU.
---

# UMA-Inverse — served inference

Ligand-conditioned protein **inverse folding**: given a fixed protein(-ligand)
backbone, the model designs an amino-acid sequence, with per-residue confidence.

This Space is a **CPU demo** (HF Spaces CPU Basic, 2 vCPU). Autoregressive
decoding runs the decoder once per residue, so latency scales with structure
size — small structures (≤ 120 residues) return in seconds to tens of seconds;
larger ones are rejected with HTTP 413. Bundled examples may be served from
precomputed results so the demo stays snappy. For production-scale workloads
you would run the same image on a GPU endpoint.

## Interfaces

- **UI:** the Gradio app at `/` (Design · Live metrics · API/Agent usage).
- **REST:** `POST /design`, `GET /health`, `GET /metrics`, OpenAPI at `/docs`.
- **Agent:** an MCP tool wraps `/design` — see the source repo's `src/mcp/`.

The model weights (`WSobo/Uma-Inverse-1`) are fetched from the Hugging Face Hub
at container startup, not baked into the image.

> This README's front-matter is what HF Spaces needs. To deploy, push this
> directory's contents to a Hugging Face Space (Docker SDK); the weights are
> pulled from the Hub at startup.
