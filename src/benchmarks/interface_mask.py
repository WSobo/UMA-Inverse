"""Sidechain-to-ligand interface mask — matches the LigandMPNN paper's residue
selection criterion.

The stock parser (``src.data.pdb_parser``) retains only the four backbone atoms
(N, CA, C, O) per residue, which is sufficient for featurisation but cannot
support the "sidechain atoms within 5.0 Å of any nonprotein atoms" definition
used by Dauparas et al. for interface sequence-recovery evaluation. This module
re-parses the PDB with BioPython to compute that mask directly on heavy-atom
coordinates, aligned with the residue ordering of a :class:`StructureContext`.

The cost is one extra PDB read per evaluated structure (cheap compared to
encoder runtime) and a clean apples-to-apples comparison with the paper.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Atoms that are NOT sidechain. Glycine has no sidechain heavy atoms; by this
# strict definition glycines never appear in the interface mask.
_BACKBONE_ATOM_NAMES: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})
_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "H2O", "DOD", "D2O", "DIS"})
_HYDROGEN_ELEMENTS: frozenset[str] = frozenset({"H", "D"})


def _build_residue_key(chain_id: str, res_num: int, icode: str) -> str:
    """Match the residue_id string format used by InferenceSession."""
    icode = icode.strip()
    return f"{chain_id}{res_num}{icode}" if icode else f"{chain_id}{res_num}"


def compute_sidechain_interface_mask(
    pdb_path: str | Path,
    residue_ids: list[str],
    *,
    cutoff: float = 5.0,
    include_zero_occupancy: bool = False,
) -> torch.Tensor:
    """Return a ``[L]`` bool mask keyed to *residue_ids*.

    ``True`` for residues whose sidechain has at least one heavy atom within
    *cutoff* Å of any nonprotein heavy atom (skipping waters and hydrogens).
    Matches the LigandMPNN paper's interface definition.

    Glycines have no sidechain heavy atoms and are always ``False``.

    Args:
        pdb_path: Path to the source PDB.
        residue_ids: Ordered residue IDs from ``StructureContext.residue_ids``
            (strings like ``"A23"`` or ``"B42C"``). The returned tensor is
            aligned to this list.
        cutoff: Distance threshold in Å (default 5.0).
        include_zero_occupancy: Mirror the stock parser flag — keep atoms with
            occupancy 0 when True.

    Returns:
        ``[L]`` bool tensor (CPU). ``False`` for residues with no sidechain
        heavy atoms, or whose nearest ligand distance exceeds *cutoff*, or that
        cannot be resolved in the PDB.
    """
    try:
        from Bio.PDB import PDBParser as _BPParser
    except ImportError as exc:
        raise ImportError(
            "biopython is required for interface mask computation"
        ) from exc

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    structure = _BPParser(QUIET=True).get_structure("s", str(pdb_path))
    model = next(iter(structure))

    # (1) Collect ligand heavy-atom coordinates from all chains.
    ligand_coords: list[list[float]] = []
    # (2) Collect per-residue sidechain coordinates keyed by residue_id.
    sidechain_by_id: dict[str, list[list[float]]] = {}

    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            het_flag, res_num, icode = residue.get_id()
            resname = residue.get_resname().strip().upper()

            if het_flag == " ":
                # Standard protein residue — harvest sidechain heavy atoms.
                rid = _build_residue_key(chain_id, int(res_num), icode)
                side: list[list[float]] = []
                for atom in residue.get_atoms():
                    if not include_zero_occupancy and atom.get_occupancy() == 0.0:
                        continue
                    if atom.get_name() in _BACKBONE_ATOM_NAMES:
                        continue
                    elem = (atom.element or atom.get_name()[0]).strip().upper()
                    if elem in _HYDROGEN_ELEMENTS:
                        continue
                    side.append(list(atom.get_coord()))
                if side:
                    sidechain_by_id[rid] = side
            elif het_flag == "W":
                continue
            else:
                # HETATM — treat as ligand unless it's a water name slipping
                # through (BioPython sometimes sets het_flag for waters).
                if resname in _WATER_RESNAMES:
                    continue
                for atom in residue.get_atoms():
                    if not include_zero_occupancy and atom.get_occupancy() == 0.0:
                        continue
                    elem = (atom.element or atom.get_name()[0]).strip().upper()
                    if elem in _HYDROGEN_ELEMENTS:
                        continue
                    ligand_coords.append(list(atom.get_coord()))

    L = len(residue_ids)
    mask = torch.zeros(L, dtype=torch.bool)

    if not ligand_coords:
        logger.debug("%s: no ligand atoms found — interface mask is all False", pdb_path)
        return mask

    ligand_t = torch.tensor(ligand_coords, dtype=torch.float32)  # [M, 3]

    for i, rid in enumerate(residue_ids):
        side = sidechain_by_id.get(rid)
        if not side:
            continue
        side_t = torch.tensor(side, dtype=torch.float32)  # [K, 3]
        # Min distance from any sidechain atom to any ligand atom
        d = torch.cdist(side_t, ligand_t).min()
        if d.item() <= cutoff:
            mask[i] = True

    return mask
