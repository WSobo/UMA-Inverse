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


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
