"""FastMCP server: the deployed UMA-Inverse model as an agent tool.

Exposes one tool, ``design_sequence_for_structure``, that POSTs a PDB to the
deployed ``/design`` endpoint and returns a **markdown** result (not raw JSON) —
matching the genesis-bio-mcp house rule that tool output should be markdown for
LLM grounding.

Configure the target endpoint with ``UMA_API_URL`` (default the local server).
Run with::

    UMA_API_URL=https://<user>-uma-inverse.hf.space uv run python -m src.mcp.server
"""
from __future__ import annotations

import os

import httpx
from fastmcp import FastMCP

API_URL = os.environ.get("UMA_API_URL", "http://127.0.0.1:7860").rstrip("/")
HTTP_TIMEOUT_S = float(os.environ.get("UMA_MCP_TIMEOUT_S", "320"))

mcp = FastMCP("uma-inverse")


@mcp.tool
def design_sequence_for_structure(
    pdb: str,
    ligand: str | None = None,
    temperature: float = 0.1,
    n_samples: int = 1,
) -> str:
    """Design an amino-acid sequence for a protein(-ligand) backbone.

    Calls the deployed UMA-Inverse inverse-folding model. The structure must be
    provided as full PDB text; any ligand is read from the PDB's HETATM records.
    This is a CPU-served demo, so keep structures small (≤ ~250 residues).

    Args:
        pdb: Full PDB file contents as text.
        ligand: Reserved; the model conditions on ligand atoms parsed from the PDB.
        temperature: Sampling temperature (0.0 = most likely residue).
        n_samples: Number of independent sequences to design.

    Returns:
        A markdown summary: designed sequence(s), mean confidence, residue count,
        and inference latency.
    """
    return design_via_http(pdb, ligand=ligand, temperature=temperature, n_samples=n_samples)


def design_via_http(
    pdb: str,
    ligand: str | None = None,
    temperature: float = 0.1,
    n_samples: int = 1,
) -> str:
    """Core logic behind the tool (decorator-free, so it's directly testable)."""
    payload = {
        "pdb": pdb,
        "ligand": ligand,
        "temperature": temperature,
        "n_samples": n_samples,
    }
    try:
        resp = httpx.post(f"{API_URL}/design", json=payload, timeout=HTTP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return f"**Error contacting UMA-Inverse** at `{API_URL}`: {exc}"

    if resp.status_code == 413:
        return (
            "**Structure too large for the CPU demo.** "
            f"{_detail(resp)} Try a smaller structure (≤ ~120 residues)."
        )
    if resp.status_code == 422:
        return f"**Invalid request.** {_detail(resp)}"
    if resp.status_code == 504:
        return f"**Timed out.** {_detail(resp)}"
    if resp.status_code != 200:
        return f"**UMA-Inverse error ({resp.status_code}).** {_detail(resp)}"

    data = resp.json()
    return _format_markdown(data)


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except Exception:  # noqa: BLE001
        return resp.text


def _format_markdown(data: dict) -> str:
    seqs = data.get("sequences", [])
    confs = data.get("per_residue_confidence", [[]])
    lines = [
        "## UMA-Inverse design",
        "",
        f"- **Residues:** {data.get('n_residues', '?')}",
        f"- **Mean confidence:** {data.get('mean_confidence', float('nan')):.3f}",
        f"- **Inference time:** {data.get('inference_ms', float('nan')):.0f} ms",
        f"- **Request id:** `{data.get('request_id', '?')}`",
        "",
        "### Designed sequence(s)",
    ]
    for i, seq in enumerate(seqs):
        per_res = confs[i] if i < len(confs) else []
        mean_c = (sum(per_res) / len(per_res)) if per_res else float("nan")
        lines.append(f"**Sample {i}** (mean per-residue confidence {mean_c:.3f}):")
        lines.append("```")
        lines.append(seq)
        lines.append("```")
    return "\n".join(lines)


@mcp.tool
def score_structure(
    pdb: str,
    sequence: str | None = None,
    mode: str = "autoregressive",
) -> str:
    """Score a protein(-ligand) structure's sequence under UMA-Inverse.

    Returns how well a sequence fits a fixed backbone: per-residue likelihoods,
    overall perplexity, recovery, and a table of **candidate mutations** — the
    most "surprising" residues where the model prefers a different amino acid.
    Useful for downstream insight (flag suboptimal residues, propose mutations).
    The structure is full PDB text; ligand atoms are read from HETATM records.

    Args:
        pdb: Full PDB file contents as text.
        sequence: Optional one-letter AA sequence to score (must match the parsed
            residue count). Defaults to the structure's native sequence.
        mode: "autoregressive" (fast) or "single-aa" (per-residue, slower).

    Returns:
        A markdown summary with a candidate-mutation table.
    """
    return score_via_http(pdb, sequence=sequence, mode=mode)


def score_via_http(pdb: str, sequence: str | None = None, mode: str = "autoregressive") -> str:
    """Core logic behind the score tool (decorator-free, so it's directly testable)."""
    payload = {"pdb": pdb, "sequence": sequence, "mode": mode}
    try:
        resp = httpx.post(f"{API_URL}/score", json=payload, timeout=HTTP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return f"**Error contacting UMA-Inverse** at `{API_URL}`: {exc}"

    if resp.status_code == 413:
        return (
            "**Structure too large for the CPU demo.** "
            f"{_detail(resp)} Try a smaller structure."
        )
    if resp.status_code in (400, 422):
        return f"**Invalid request.** {_detail(resp)}"
    if resp.status_code == 504:
        return f"**Timed out.** {_detail(resp)}"
    if resp.status_code != 200:
        return f"**UMA-Inverse error ({resp.status_code}).** {_detail(resp)}"

    return _format_score_markdown(resp.json())


def _format_score_markdown(data: dict) -> str:
    positions = data.get("positions", [])
    lines = [
        "## UMA-Inverse score",
        "",
        f"- **Residues:** {data.get('n_residues', '?')}",
        f"- **Perplexity:** {data.get('perplexity', float('nan')):.2f} "
        "(lower = the sequence fits the structure better; ~20 = random)",
        f"- **Mean log-likelihood:** {data.get('mean_log_prob', float('nan')):.3f}",
        f"- **Recovery (model top-1 == sequence):** {data.get('recovery', float('nan')) * 100:.0f}%",
        f"- **Inference time:** {data.get('inference_ms', float('nan')):.0f} ms",
        f"- **Request id:** `{data.get('request_id', '?')}`",
        "",
    ]
    # Candidate mutations: positions where the model prefers a different residue,
    # ranked by how unlikely the current residue is (lowest log-prob first).
    candidates = [p for p in positions if p.get("top_aa") != p.get("aa")]
    candidates.sort(key=lambda p: p.get("log_prob", 0.0))
    top = candidates[:10]
    if top:
        lines.append("### Candidate mutations (most 'surprising' residues)")
        lines.append("")
        lines.append("| residue | current | log-prob | model prefers | prob |")
        lines.append("|---|---|---|---|---|")
        for p in top:
            lines.append(
                f"| {p['residue_id']} | {p['aa']} | {p['log_prob']:.2f} | "
                f"{p['top_aa']} | {p['top_prob']:.2f} |"
            )
    else:
        lines.append("_The model's top prediction matches the sequence at every position._")
    return "\n".join(lines)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
