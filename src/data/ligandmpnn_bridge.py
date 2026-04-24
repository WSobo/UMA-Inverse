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


def _encode_ligand_atomic_numbers(y_t: torch.Tensor) -> torch.Tensor:
    """Return atomic numbers as int64 indices for an ``nn.Embedding`` layer.

    Values are in [1, 118] for real elements plus 119 for "unknown" (the
    parser's fallback for PDB element strings outside the standard table).
    Index 0 is reserved for padding by the embedding layer.
    """
    return y_t.long()


def _construct_virtual_cb(x: torch.Tensor) -> torch.Tensor:
    """Construct virtual Cβ positions from N, Cα, C using ProteinMPNN's formula.

    Coefficients match LigandMPNN/data_utils.py:821-824 and place virtual Cβ
    at ~1.52 Å from Cα — the canonical covalent C-C bond length. The
    construction is analytic, so it works for glycine (where a real Cβ
    doesn't exist) without a special case; the placed virtual Cβ is where
    Cβ *would* be if the residue weren't Gly.

    Args:
        x: ``[L, 4, 3]`` backbone coordinates in order N, Cα, C, O.

    Returns:
        ``[L, 3]`` virtual Cβ coordinates.
    """
    n = x[:, 0, :]
    ca = x[:, 1, :]
    c = x[:, 2, :]
    b = ca - n
    c_vec = c - ca
    a = torch.linalg.cross(b, c_vec, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


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

def _format_residue_id(chain_id: str, res_num: int, insertion_code: str) -> str:
    """Format a per-residue identifier matching LigandMPNN's convention.

    Format: ``<chain><resnum>[<insertion_code>]`` — e.g. ``"A23"``, ``"B42C"``.
    Insertion codes are appended verbatim when non-empty.
    """
    return f"{chain_id}{res_num}{insertion_code}"


def load_example_from_pdb(
    pdb_path: str,
    ligand_context_atoms: int = 25,
    cutoff_for_score: float = 8.0,
    max_total_nodes: int = 384,
    device: str = "cpu",
    parse_chains: Optional[List[str]] = None,
    include_zero_occupancy: bool = False,
    return_residue_ids: bool = False,
    ligand_featurizer: str = "onehot6",
    residue_anchor: str = "ca",
) -> Dict[str, object]:
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
        parse_chains: When provided, only residues from these chain IDs are
            included. Ligand atoms are still parsed from every chain.
        include_zero_occupancy: When True, atoms with occupancy=0 are retained
            (matches LigandMPNN's ``parse_atoms_with_zero_occupancy=1``).
        return_residue_ids: When True, the output dict includes a
            ``residue_ids`` key — a ``List[str]`` of per-residue identifiers
            (``"A23"``, ``"B42C"``) in the same order as ``residue_coords``.
            Required for inference-time chain-letter residue selection
            (fixed/redesigned residues, per-residue bias/omit). The training
            pipeline leaves this False to avoid the small per-batch overhead.
        ligand_featurizer: ``"onehot6"`` (v1 default) or
            ``"atomic_number_embedding"`` (v2 phase 1). The onehot path emits
            a ``[M, 6]`` float tensor under key ``ligand_features``; the
            embedding path emits ``[M]`` int64 atomic numbers under key
            ``ligand_atomic_numbers``. Mutually exclusive — only one of the
            two keys is present in the returned dict.
        residue_anchor: ``"ca"`` (v1 default) or ``"cb"`` (v2 phase 2).
            Controls which atom is used as the per-residue anchor for
            ``residue_coords``. ``"ca"`` emits the Cα position unchanged;
            ``"cb"`` emits a virtual Cβ constructed analytically from
            N/Cα/C — places the residue anchor closer to the sidechain,
            which matters for distance-based pair features between residues
            and ligand atoms.

    Returns:
        Dict with keys:

        * ``residue_coords``         ``[L, 3]``   Cα or virtual Cβ coordinates
        * ``residue_features``       ``[L, 6]``   sin/cos backbone dihedrals
        * ``residue_mask``           ``[L]``       all-True (valid residues only)
        * ``sequence``               ``[L]``       int64 AA token indices
        * ``design_mask``            ``[L]``       bool — which residues to design
        * ``ligand_coords``          ``[M, 3]``    ligand heavy-atom coords
        * ``ligand_features``        ``[M, 6]``    element one-hot (onehot6 only)
        * ``ligand_atomic_numbers``  ``[M]``        int64 atomic numbers (embedding only)
        * ``ligand_mask``            ``[M]``       all-True (valid ligand atoms only)
        * ``residue_anchor_atom``    ``str``        ``"ca"`` or ``"cb"`` (traceability)
        * ``residue_ids``            ``List[str]`` (when ``return_residue_ids=True``)
    """
    if ligand_featurizer not in ("onehot6", "atomic_number_embedding"):
        raise ValueError(
            f"unknown ligand_featurizer={ligand_featurizer!r}; "
            "expected 'onehot6' or 'atomic_number_embedding'"
        )
    if residue_anchor not in ("ca", "cb"):
        raise ValueError(
            f"unknown residue_anchor={residue_anchor!r}; expected 'ca' or 'cb'"
        )
    parsed = parse_pdb(
        pdb_path,
        cutoff_for_score=cutoff_for_score,
        device=device,
        parse_chains=parse_chains,
        include_zero_occupancy=include_zero_occupancy,
    )

    x            = parsed["X"]           # [L, 4, 3]
    residue_mask = parsed["mask"].bool() # [L]
    chain_mask   = parsed["chain_mask"]  # [L]

    # Per-residue anchor selection — Cα (v1) vs virtual Cβ (v2 phase 2).
    # Cβ is constructed before masking so the [L, 3] shape matches downstream
    # slicing, then masked the same way Cα was before.
    if residue_anchor == "ca":
        anchor_coords = x[:, 1, :]
    else:  # "cb" — validated at function entry
        anchor_coords = _construct_virtual_cb(x)

    # Mask down to valid (Cα-present) residues
    residue_coords   = anchor_coords[residue_mask]             # [L_valid, 3]
    residue_features = _compute_backbone_dihedrals(x)[residue_mask]  # [L_valid, 6]
    sequence         = parsed["S"][residue_mask].long()        # [L_valid]
    design_mask      = chain_mask[residue_mask].bool()         # [L_valid]

    residue_ids_valid: Optional[List[str]] = None
    if return_residue_ids:
        chain_ids = parsed["chain_ids"]
        res_nums = parsed["res_nums"]
        insertion_codes = parsed["insertion_codes"]
        mask_list = residue_mask.tolist()
        residue_ids_valid = [
            _format_residue_id(chain_ids[i], res_nums[i], insertion_codes[i])
            for i in range(len(mask_list))
            if mask_list[i]
        ]

    y   = parsed["Y"]           # [M_all, 3]
    y_t = parsed["Y_t"]         # [M_all]
    y_m = parsed["Y_m"].bool()  # [M_all]

    ligand_coords   = y[y_m]    # [M_near, 3]
    ligand_elements = y_t[y_m]  # [M_near]

    if ligand_coords.numel() == 0:
        ligand_coords = torch.zeros((0, 3), dtype=torch.float32, device=device)
        ligand_elements = torch.zeros((0,), dtype=torch.long, device=device)
    else:
        # Keep nearest ligand atoms to residue centroid within fixed memory budget
        if ligand_coords.shape[0] > ligand_context_atoms:
            residue_center = residue_coords.mean(dim=0, keepdim=True)
            dist = torch.linalg.norm(ligand_coords - residue_center, dim=-1)
            keep = torch.topk(dist, k=ligand_context_atoms, largest=False).indices
            ligand_coords   = ligand_coords[keep]
            ligand_elements = ligand_elements[keep]

    if ligand_featurizer == "onehot6":
        ligand_repr_tensor = _encode_ligand_elements(ligand_elements)
    else:  # "atomic_number_embedding" — validated at function entry
        ligand_repr_tensor = _encode_ligand_atomic_numbers(ligand_elements)

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

    if residue_ids_valid is not None:
        keep_indices = keep_idx.tolist()
        residue_ids_valid = [residue_ids_valid[i] for i in keep_indices]

    output: Dict[str, object] = {
        "residue_coords":      residue_coords.float(),
        "residue_features":    residue_features.float(),
        "residue_mask":        torch.ones(residue_coords.shape[0], dtype=torch.bool, device=device),
        "sequence":            sequence.long(),
        "design_mask":         design_mask.bool(),
        "ligand_coords":       ligand_coords.float(),
        "ligand_mask":         torch.ones(ligand_coords.shape[0], dtype=torch.bool, device=device),
        # Traceability — downstream code can verify which anchor the coords
        # came from without re-reading the config. Not a tensor, so it is
        # dropped by collate_batch (which only stacks known tensor keys).
        "residue_anchor_atom": residue_anchor,
    }
    if ligand_featurizer == "onehot6":
        output["ligand_features"] = ligand_repr_tensor.float()
    else:  # "atomic_number_embedding"
        output["ligand_atomic_numbers"] = ligand_repr_tensor.long()
    if residue_ids_valid is not None:
        output["residue_ids"] = residue_ids_valid
    return output
