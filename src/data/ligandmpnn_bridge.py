"""Bridge between UMA-Inverse and LigandMPNN-style PDB data.

Previously imported ``parse_PDB`` and ``featurize`` directly from the
``../LigandMPNN/`` reference repo at runtime.  That dependency has been
replaced by the self-contained ``src.data.pdb_parser`` module, so cloning
LigandMPNN is no longer required to run the model — only its training split
JSON files (``train.json``, ``valid.json``) are still needed.
"""
import json
import logging
import os
from typing import Dict, List, Optional

import torch

from .pdb_parser import parse_pdb

logger = logging.getLogger(__name__)


def load_json_ids(json_path: str) -> List[str]:
    """Load a list of PDB IDs from a LigandMPNN-style JSON split file."""
    with open(json_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return [str(x) for x in ids]


def resolve_pdb_path(pdb_dir: str, pdb_id: str) -> Optional[str]:
    """Return the first existing path for *pdb_id* under *pdb_dir*, or None."""
    pdb_id = str(pdb_id)
    sub_dir = pdb_id.lower()[1:3] if len(pdb_id) >= 4 else "misc"
    candidates = [
        os.path.join(pdb_dir, sub_dir, f"{pdb_id}.pdb"),
        os.path.join(pdb_dir, sub_dir, f"{pdb_id}.pt"),
        os.path.join(pdb_dir, f"{pdb_id}.pdb"),
        os.path.join(pdb_dir, f"{pdb_id}.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ── Feature engineering helpers ───────────────────────────────────────────────

def _compute_backbone_dihedrals(x: torch.Tensor) -> torch.Tensor:
    """Compute sin/cos of backbone dihedral angles (φ, ψ, ω).

    Args:
        x: ``[L, 4, 3]`` backbone atom coordinates in order N, Cα, C, O.

    Returns:
        ``[L, 6]`` tensor — [sin φ, sin ψ, sin ω, cos φ, cos ψ, cos ω].
    """
    n  = x[:, 0, :]
    ca = x[:, 1, :]
    c  = x[:, 2, :]

    c_prev  = torch.cat([c[:1],  c[:-1]],  dim=0)
    n_next  = torch.cat([n[1:],  n[-1:]],  dim=0)
    ca_next = torch.cat([ca[1:], ca[-1:]], dim=0)

    def _dihedral(p0: torch.Tensor, p1: torch.Tensor,
                  p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + 1e-8)
        n1 = torch.linalg.cross(b0, b1_norm, dim=-1)
        n2 = torch.linalg.cross(b1_norm, b2, dim=-1)
        m1 = torch.linalg.cross(n1, b1_norm, dim=-1)
        return torch.atan2((m1 * n2).sum(dim=-1), (n1 * n2).sum(dim=-1))

    phi   = _dihedral(c_prev, n,  ca, c)
    psi   = _dihedral(n,  ca,  c,  n_next)
    omega = _dihedral(ca, c,   n_next, ca_next)

    angles = torch.stack([phi, psi, omega], dim=-1)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _encode_ligand_elements(y_t: torch.Tensor) -> torch.Tensor:
    """One-hot encode atomic numbers into 6 element bins: C, N, O, S, P, other.

    Args:
        y_t: ``[M]`` int tensor of atomic numbers.

    Returns:
        ``[M, 6]`` float32 one-hot tensor.
    """
    out = torch.zeros((y_t.shape[0], 6), dtype=torch.float32, device=y_t.device)
    out[y_t == 6,  0] = 1.0   # C
    out[y_t == 7,  1] = 1.0   # N
    out[y_t == 8,  2] = 1.0   # O
    out[y_t == 16, 3] = 1.0   # S
    out[y_t == 15, 4] = 1.0   # P
    other = ~((y_t == 6) | (y_t == 7) | (y_t == 8) | (y_t == 16) | (y_t == 15))
    out[other, 5] = 1.0
    return out


def _select_residue_crop(
    residue_coords: torch.Tensor,
    ligand_coords: torch.Tensor,
    max_residues: int,
) -> torch.Tensor:
    """Return indices of the *max_residues* residues closest to the ligand centroid.

    Falls back to leading residues when no ligand atoms are present.
    """
    if residue_coords.shape[0] <= max_residues:
        return torch.arange(residue_coords.shape[0], device=residue_coords.device)

    if ligand_coords.shape[0] == 0:
        return torch.arange(max_residues, device=residue_coords.device)

    ligand_center = ligand_coords.mean(dim=0, keepdim=True)
    dist = torch.linalg.norm(residue_coords - ligand_center, dim=-1)
    topk = torch.topk(dist, k=max_residues, largest=False).indices
    topk, _ = torch.sort(topk)
    return topk


# ── Public entry point ────────────────────────────────────────────────────────

def load_example_from_pdb(
    pdb_path: str,
    ligand_context_atoms: int = 25,
    cutoff_for_score: float = 8.0,
    max_total_nodes: int = 384,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Featurize a single PDB file into model-ready tensors.

    Uses the self-contained ``pdb_parser.parse_pdb`` — no LigandMPNN clone
    required.

    Args:
        pdb_path: Path to a ``.pdb`` file.
        ligand_context_atoms: Maximum number of ligand atoms to keep (nearest
            to the residue centroid).
        cutoff_for_score: Distance threshold (Å) passed to the parser for
            marking ligand atoms that are near the protein.
        max_total_nodes: Hard cap on total nodes (residues + ligand atoms).
            Excess residues are cropped to those nearest the ligand centroid.
        device: Torch device string.

    Returns:
        Dict with keys:

        * ``residue_coords``   ``[L, 3]``   Cα coordinates
        * ``residue_features`` ``[L, 6]``   sin/cos backbone dihedrals
        * ``residue_mask``     ``[L]``       all-True (valid residues only)
        * ``sequence``         ``[L]``       int64 AA token indices
        * ``design_mask``      ``[L]``       bool — which residues to design
        * ``ligand_coords``    ``[M, 3]``    ligand heavy-atom coords
        * ``ligand_features``  ``[M, 6]``    element one-hot
        * ``ligand_mask``      ``[M]``       all-True (valid ligand atoms only)
    """
    parsed = parse_pdb(pdb_path, cutoff_for_score=cutoff_for_score, device=device)

    x            = parsed["X"]           # [L, 4, 3]
    residue_mask = parsed["mask"].bool() # [L]
    chain_mask   = parsed["chain_mask"]  # [L]

    # Mask down to valid (Cα-present) residues
    residue_coords   = x[:, 1, :][residue_mask]               # [L_valid, 3]
    residue_features = _compute_backbone_dihedrals(x)[residue_mask]  # [L_valid, 6]
    sequence         = parsed["S"][residue_mask].long()        # [L_valid]
    design_mask      = chain_mask[residue_mask].bool()         # [L_valid]

    y   = parsed["Y"]           # [M_all, 3]
    y_t = parsed["Y_t"]         # [M_all]
    y_m = parsed["Y_m"].bool()  # [M_all]

    ligand_coords   = y[y_m]    # [M_near, 3]
    ligand_elements = y_t[y_m]  # [M_near]

    if ligand_coords.numel() == 0:
        ligand_coords   = torch.zeros((0, 3), dtype=torch.float32, device=device)
        ligand_features = torch.zeros((0, 6), dtype=torch.float32, device=device)
    else:
        # Keep nearest ligand atoms to residue centroid within fixed memory budget
        if ligand_coords.shape[0] > ligand_context_atoms:
            residue_center = residue_coords.mean(dim=0, keepdim=True)
            dist = torch.linalg.norm(ligand_coords - residue_center, dim=-1)
            keep = torch.topk(dist, k=ligand_context_atoms, largest=False).indices
            ligand_coords   = ligand_coords[keep]
            ligand_elements = ligand_elements[keep]
        ligand_features = _encode_ligand_elements(ligand_elements)

    max_residues = max(1, max_total_nodes - ligand_coords.shape[0])
    keep_idx = _select_residue_crop(
        residue_coords=residue_coords,
        ligand_coords=ligand_coords,
        max_residues=max_residues,
    )

    n_cropped = residue_coords.shape[0] - keep_idx.shape[0]
    if n_cropped > 0:
        logger.debug(
            "%s: cropped %d/%d residues to stay within max_total_nodes=%d",
            pdb_path,
            n_cropped,
            residue_coords.shape[0],
            max_total_nodes,
        )

    residue_coords   = residue_coords[keep_idx]
    residue_features = residue_features[keep_idx]
    sequence         = sequence[keep_idx]
    design_mask      = design_mask[keep_idx]

    return {
        "residue_coords":   residue_coords.float(),
        "residue_features": residue_features.float(),
        "residue_mask":     torch.ones(residue_coords.shape[0], dtype=torch.bool, device=device),
        "sequence":         sequence.long(),
        "design_mask":      design_mask.bool(),
        "ligand_coords":    ligand_coords.float(),
        "ligand_features":  ligand_features.float(),
        "ligand_mask":      torch.ones(ligand_coords.shape[0], dtype=torch.bool, device=device),
    }
