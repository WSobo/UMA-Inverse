"""Gradio dashboard for the UMA-Inverse Space.

Three tabs:
  1. Design       — paste/upload a PDB, design sequences, see a per-residue
                    confidence plot and the request latency.
  2. Live metrics — p50/p90/p99 latency, confidence distribution, and request
                    counts, parsed from the service's own Prometheus metrics.
  3. API / Agent  — the REST contract, a curl example, and the MCP usage.

The design tab calls the in-process inference adapter directly and records the
same metrics the REST endpoint does, so the live-metrics tab reflects UI usage
too. Bundled examples prefer a precomputed result (instant) when one is shipped
alongside the PDB, falling back to live inference for small structures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering for the Space
import gradio as gr
import matplotlib.pyplot as plt
from prometheus_client.parser import text_string_to_metric_families

from src.serving import metrics as M
from src.serving.inference import DEFAULT_MAX_RESIDUES, run_inference, score_inference

EXAMPLES_DIR = Path(__file__).parent / "examples"
SELF_URL = os.environ.get("UMA_SELF_URL", "http://127.0.0.1:7860")


# ── Example discovery ──────────────────────────────────────────────────────────


def _example_labels() -> dict[str, Path]:
    """Map a friendly label → example PDB path."""
    if not EXAMPLES_DIR.is_dir():
        return {}
    return {p.stem.upper(): p for p in sorted(EXAMPLES_DIR.glob("*.pdb"))}


def _precomputed_result(pdb_path: Path) -> dict | None:
    """Load a shipped precomputed result JSON (``<id>.result.json``), if present."""
    cand = pdb_path.with_suffix(".result.json")
    if cand.exists():
        try:
            return json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _load_example(label: str) -> str:
    examples = _example_labels()
    path = examples.get(label)
    if path is None:
        return ""
    return path.read_text(encoding="utf-8")


# ── Plotting ────────────────────────────────────────────────────────────────────


def _confidence_figure(per_residue: list[float], title: str):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    xs = list(range(len(per_residue)))
    ax.bar(xs, per_residue, width=1.0, color="#2b8cbe")
    ax.axhline(sum(per_residue) / max(len(per_residue), 1), color="#e34a33",
               linestyle="--", linewidth=1, label="mean")
    ax.set_xlabel("residue index")
    ax.set_ylabel("confidence")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ── Tab 1: Design ───────────────────────────────────────────────────────────────


def _design_fn(pdb_text: str, pdb_file, ligand: str, temperature: float, n_samples: int):
    pdb_str = ""
    if pdb_file is not None:
        try:
            pdb_str = Path(pdb_file).read_text(encoding="utf-8")
        except (OSError, TypeError):
            pdb_str = ""
    if not pdb_str:
        pdb_str = (pdb_text or "").strip()

    if not pdb_str:
        return "⚠️ Provide a PDB (paste text or upload a file).", None, ""

    # If this exact PDB matches a bundled example with a precomputed result,
    # serve that instantly (keeps the demo snappy for larger ligand examples).
    for path in _example_labels().values():
        if path.read_text(encoding="utf-8").strip() == pdb_str:
            pre = _precomputed_result(path)
            if pre is not None:
                fig = _confidence_figure(
                    pre["per_residue_confidence"][0], f"{path.stem.upper()} (precomputed)"
                )
                seqs = "\n".join(f"> sample {i}\n{s}" for i, s in enumerate(pre["sequences"]))
                meta = (
                    f"**Precomputed** · {pre['n_residues']} residues · "
                    f"mean confidence {pre['mean_confidence']:.3f} · "
                    f"{pre['inference_ms']:.0f} ms (original run)"
                )
                return f"```\n{seqs}\n```", fig, meta
            break

    try:
        result = run_inference(
            pdb_str,
            ligand=(ligand or None),
            temperature=float(temperature),
            n_samples=int(n_samples),
        )
    except Exception as exc:  # noqa: BLE001 — surface errors to the UI cleanly
        return f"❌ {type(exc).__name__}: {exc}", None, ""

    M.record_design_metrics(
        n_residues=result.n_residues,
        mean_confidence=result.mean_confidence,
        inference_ms=result.inference_ms,
    )

    fig = _confidence_figure(result.per_residue_confidence[0], "per-residue confidence (sample 0)")
    seqs = "\n".join(f"> sample {i}\n{s}" for i, s in enumerate(result.sequences))
    meta = (
        f"{result.n_residues} residues · mean confidence "
        f"{result.mean_confidence:.3f} · {result.inference_ms:.0f} ms"
    )
    return f"```\n{seqs}\n```", fig, meta


# ── Tab: Score ───────────────────────────────────────────────────────────────


def _read_pdb(pdb_file, pdb_text: str) -> str:
    if pdb_file is not None:
        try:
            return Path(pdb_file).read_text(encoding="utf-8")
        except (OSError, TypeError):
            pass
    return (pdb_text or "").strip()


def _score_figure(positions: list):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    xs = [p.position for p in positions]
    ys = [p.log_prob for p in positions]
    colors = ["#e34a33" if p.top_aa != p.aa else "#2b8cbe" for p in positions]
    ax.bar(xs, ys, width=1.0, color=colors)
    ax.set_xlabel("residue index")
    ax.set_ylabel("log-likelihood")
    ax.set_title("per-residue log-likelihood (red = model prefers a different residue)")
    fig.tight_layout()
    return fig


def _score_fn(pdb_text: str, pdb_file, sequence: str, mode: str):
    pdb_str = _read_pdb(pdb_file, pdb_text)
    if not pdb_str:
        return "⚠️ Provide a PDB (paste text or upload a file).", None, []
    seq = (sequence or "").strip() or None
    try:
        result = score_inference(pdb_str, sequence=seq, mode=mode)
    except Exception as exc:  # noqa: BLE001 — surface errors to the UI cleanly
        return f"❌ {type(exc).__name__}: {exc}", None, []

    M.record_score_metrics(
        n_residues=result.n_residues,
        perplexity=result.perplexity,
        inference_ms=result.inference_ms,
    )
    fig = _score_figure(result.positions)
    summary = (
        f"**Perplexity:** {result.perplexity:.2f} (lower = sequence fits structure better) · "
        f"**recovery:** {result.recovery * 100:.0f}% · "
        f"**{result.n_residues} residues** · {result.inference_ms:.0f} ms · mode={result.mode}"
    )
    candidates = sorted(
        (p for p in result.positions if p.top_aa != p.aa), key=lambda p: p.log_prob
    )[:15]
    rows = [
        [p.residue_id, p.aa, round(p.log_prob, 2), p.top_aa, round(p.top_prob, 2)]
        for p in candidates
    ]
    return summary, fig, rows


# ── Tab 2: Live metrics ──────────────────────────────────────────────────────────


def _fetch_metrics_text() -> str:
    """Prefer the live /metrics endpoint; fall back to in-process rendering."""
    try:
        import httpx

        resp = httpx.get(f"{SELF_URL}/metrics", timeout=3.0)
        if resp.status_code == 200:
            return resp.text
    except Exception:  # noqa: BLE001 — fall back to in-process
        pass
    body, _ = M.render_metrics()
    return body.decode("utf-8")


def _histogram_quantile(buckets: list[tuple[float, float]], total: float, q: float) -> float | None:
    """Estimate a quantile from cumulative Prometheus histogram buckets.

    ``buckets`` is a sorted list of ``(le, cumulative_count)``. Returns the
    upper bucket boundary the quantile falls in (the standard, if coarse,
    histogram_quantile approximation).
    """
    if total <= 0 or not buckets:
        return None
    rank = q * total
    for le, cum in buckets:
        if cum >= rank:
            return le
    return buckets[-1][0]


def _parse_metrics(text: str) -> dict:
    out: dict = {
        "requests": {},
        "latency_buckets": [],
        "latency_count": 0.0,
        "latency_sum": 0.0,
        "confidence_buckets": [],
        "confidence_count": 0.0,
        "model_load_seconds": None,
        "inflight": 0.0,
    }
    for family in text_string_to_metric_families(text):
        if family.name == "uma_requests":  # counter exposes uma_requests_total
            for s in family.samples:
                if s.name == "uma_requests_total":
                    key = f"{s.labels.get('endpoint', '?')} [{s.labels.get('status', '?')}]"
                    out["requests"][key] = out["requests"].get(key, 0.0) + s.value
        elif family.name == "uma_inference_latency_seconds":
            for s in family.samples:
                if s.name.endswith("_bucket"):
                    le = s.labels.get("le")
                    if le and le != "+Inf":
                        out["latency_buckets"].append((float(le), s.value))
                elif s.name.endswith("_count"):
                    out["latency_count"] = s.value
                elif s.name.endswith("_sum"):
                    out["latency_sum"] = s.value
        elif family.name == "uma_mean_confidence":
            for s in family.samples:
                if s.name.endswith("_bucket"):
                    le = s.labels.get("le")
                    if le and le != "+Inf":
                        out["confidence_buckets"].append((float(le), s.value))
                elif s.name.endswith("_count"):
                    out["confidence_count"] = s.value
        elif family.name == "uma_model_load_seconds":
            for s in family.samples:
                out["model_load_seconds"] = s.value
        elif family.name == "uma_inflight_requests":
            for s in family.samples:
                out["inflight"] = s.value
    out["latency_buckets"].sort()
    out["confidence_buckets"].sort()
    return out


def _metrics_dashboard():
    parsed = _parse_metrics(_fetch_metrics_text())

    total_req = sum(parsed["requests"].values())
    lat = parsed["latency_buckets"]
    n = parsed["latency_count"]
    p50 = _histogram_quantile(lat, n, 0.50)
    p90 = _histogram_quantile(lat, n, 0.90)
    p99 = _histogram_quantile(lat, n, 0.99)
    avg = (parsed["latency_sum"] / n) if n else None

    def _fmt(v):
        return f"{v:.2f}s" if v is not None else "—"

    summary = (
        f"### Service health\n"
        f"- **Total requests:** {int(total_req)}\n"
        f"- **Design requests (latency-sampled):** {int(n)}\n"
        f"- **In-flight:** {int(parsed['inflight'])}\n"
        f"- **Model load time:** {_fmt(parsed['model_load_seconds'])}\n\n"
        f"### Inference latency (from real requests)\n"
        f"- **avg:** {_fmt(avg)} · **p50:** {_fmt(p50)} · "
        f"**p90:** {_fmt(p90)} · **p99:** {_fmt(p99)}\n"
    )

    # Latency percentile bar chart.
    fig_lat, ax = plt.subplots(figsize=(5, 3))
    labels = ["avg", "p50", "p90", "p99"]
    vals = [v or 0.0 for v in (avg, p50, p90, p99)]
    ax.bar(labels, vals, color="#2b8cbe")
    ax.set_ylabel("seconds")
    ax.set_title("inference latency")
    fig_lat.tight_layout()

    # Confidence distribution (incremental per-bucket counts).
    fig_conf, ax2 = plt.subplots(figsize=(5, 3))
    cb = parsed["confidence_buckets"]
    if cb:
        prev = 0.0
        xs, ys = [], []
        for le, cum in cb:
            xs.append(le)
            ys.append(max(cum - prev, 0.0))
            prev = cum
        ax2.bar([str(x) for x in xs], ys, color="#31a354")
    ax2.set_xlabel("mean confidence ≤")
    ax2.set_ylabel("count")
    ax2.set_title("confidence distribution")
    fig_conf.tight_layout()

    # Request counts table.
    req_rows = [[k, int(v)] for k, v in sorted(parsed["requests"].items())]
    return summary, fig_lat, fig_conf, req_rows


# ── Tab 3 content ────────────────────────────────────────────────────────────────

_API_DOC = f"""
## REST API contract

### `POST /design`
```json
{{
  "pdb": "<full PDB file contents as text>",
  "ligand": null,
  "temperature": 0.1,
  "n_samples": 1
}}
```
Returns `sequences`, `per_residue_confidence`, `mean_confidence`, `n_residues`,
`inference_ms`, and a `request_id`. Response headers include `X-Request-ID` and
`X-Inference-MS`. Structures larger than **{DEFAULT_MAX_RESIDUES} residues** are
rejected with `413` (this is a CPU demo).

### curl
```bash
curl -X POST "$SPACE_URL/design" \\
  -H "Content-Type: application/json" \\
  -d "{{\\"pdb\\": \\"$(cat structure.pdb | sed 's/\\"/\\\\\\"/g')\\", \\"n_samples\\": 2}}"
```

### `POST /score`
```json
{{ "pdb": "<full PDB text>", "sequence": null, "mode": "autoregressive" }}
```
Scores a sequence against the structure. Returns per-residue `log_prob`/`prob`,
the model's preferred residue (`top_aa`/`top_prob`), overall `perplexity` (lower =
better fit), `recovery`, and per-position records — so an agent can flag suboptimal
residues and propose mutations. Omit `sequence` to score the native sequence.

### Other endpoints
- `GET /health` — liveness (`status`, `model_loaded`, `uptime_s`)
- `GET /metrics` — Prometheus exposition (latency histograms, confidence, perplexity, counts)
- `GET /docs` — OpenAPI schema (the agent-readable contract)

## Agent usage (MCP)
Two MCP tools (FastMCP server in `src/mcp/`) wrap the endpoints and return markdown:
- `design_sequence_for_structure(pdb, ligand?, temperature?)` → redesign a backbone.
- `score_structure(pdb, sequence?, mode?)` → score a sequence and get a
  **candidate-mutation** table.

The story: an agent retrieves a structure (e.g. via genesis-bio-mcp), **scores** it
to find suboptimal residues, then **redesigns** it — all against this deployed model.
"""


# ── Assembly ─────────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    examples = _example_labels()
    example_labels = list(examples.keys())

    with gr.Blocks(title="UMA-Inverse") as demo:
        gr.Markdown(
            "# UMA-Inverse — ligand-conditioned inverse folding\n"
            "Design amino-acid sequences for a fixed protein(-ligand) backbone. "
            f"**CPU demo** — structures ≤ {DEFAULT_MAX_RESIDUES} residues; latency "
            "is honest CPU latency (seconds to tens of seconds)."
        )

        with gr.Tab("Design"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Provide a structure** — upload a file, paste text, or load an example.")
                    pdb_file = gr.File(
                        label="📁 Select a .pdb file",
                        file_types=[".pdb"],
                        file_count="single",
                    )
                    pdb_text = gr.Textbox(
                        label="…or paste PDB text",
                        lines=6,
                        placeholder="ATOM ... / HETATM ...",
                    )
                    ligand = gr.Textbox(label="Ligand (optional; read from PDB HETATM)", value="")
                    temperature = gr.Slider(0.0, 2.0, value=0.1, step=0.05, label="temperature")
                    n_samples = gr.Slider(1, 8, value=1, step=1, label="n_samples")
                    run_btn = gr.Button("Design sequence(s)", variant="primary")
                    if example_labels:
                        ex_dropdown = gr.Dropdown(
                            example_labels, label="Load bundled example", value=None
                        )
                with gr.Column():
                    out_seq = gr.Markdown(label="Designed sequence(s)")
                    out_plot = gr.Plot(label="per-residue confidence")
                    out_meta = gr.Markdown()

            run_btn.click(
                _design_fn,
                inputs=[pdb_text, pdb_file, ligand, temperature, n_samples],
                outputs=[out_seq, out_plot, out_meta],
                concurrency_limit=1,  # serialise inference on the CPU box
            )
            if example_labels:
                ex_dropdown.change(_load_example, inputs=[ex_dropdown], outputs=[pdb_text])

        with gr.Tab("Score"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "**Score a sequence against its structure** — per-residue "
                        "likelihood + candidate mutations (positions the model would change)."
                    )
                    s_pdb_file = gr.File(
                        label="📁 Select a .pdb file", file_types=[".pdb"], file_count="single"
                    )
                    s_pdb_text = gr.Textbox(label="…or paste PDB text", lines=6)
                    s_sequence = gr.Textbox(
                        label="Sequence to score (optional; defaults to the native sequence)",
                        value="",
                    )
                    s_mode = gr.Dropdown(
                        ["autoregressive", "single-aa"], value="autoregressive", label="mode"
                    )
                    s_btn = gr.Button("Score", variant="primary")
                    if example_labels:
                        s_ex = gr.Dropdown(example_labels, label="Load bundled example", value=None)
                with gr.Column():
                    s_summary = gr.Markdown()
                    s_plot = gr.Plot(label="per-residue log-likelihood")
                    s_table = gr.Dataframe(
                        headers=["residue", "current", "log-prob", "prefers", "prob"],
                        label="candidate mutations",
                        interactive=False,
                    )
            s_btn.click(
                _score_fn,
                inputs=[s_pdb_text, s_pdb_file, s_sequence, s_mode],
                outputs=[s_summary, s_plot, s_table],
                concurrency_limit=1,
            )
            if example_labels:
                s_ex.change(_load_example, inputs=[s_ex], outputs=[s_pdb_text])

        with gr.Tab("Live metrics"):
            refresh = gr.Button("Refresh metrics")
            metrics_md = gr.Markdown()
            with gr.Row():
                lat_plot = gr.Plot(label="latency")
                conf_plot = gr.Plot(label="confidence")
            req_table = gr.Dataframe(
                headers=["endpoint [status]", "count"], label="request counts", interactive=False
            )
            refresh.click(
                _metrics_dashboard, outputs=[metrics_md, lat_plot, conf_plot, req_table]
            )
            # No demo.load() auto-refresh: an on-load event runs through the queue
            # and can leave the whole page in a "loading" state. Metrics refresh
            # on demand via the button instead.

        with gr.Tab("API / Agent usage"):
            gr.Markdown(_API_DOC)

    return demo
