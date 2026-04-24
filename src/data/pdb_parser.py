"""Self-contained PDB parser using BioPython.

Replaces the runtime import dependency on ../LigandMPNN/data_utils.py.
Parses backbone coordinates, sequence tokens, and ligand heavy atoms from a
PDB file using only BioPython (already a project dependency).

Output format is intentionally compatible with what ``load_example_from_pdb``
in ``ligandmpnn_bridge.py`` expects, so no downstream code changes are needed.
"""
import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ── Amino acid vocabulary ─────────────────────────────────────────────────────
# Alphabetical order of single-letter codes — MUST match LigandMPNN tokenisation.
# alphabet = 'ACDEFGHIKLMNPQRSTVWY'  (X = 20 for unknown/non-standard)
_AA3_TO_TOKEN: Dict[str, int] = {
    "ALA": 0,   # A
    "CYS": 1,   # C
    "ASP": 2,   # D
    "GLU": 3,   # E
    "PHE": 4,   # F
    "GLY": 5,   # G
    "HIS": 6,   # H
    "ILE": 7,   # I
    "LYS": 8,   # K
    "LEU": 9,   # L
    "MET": 10,  # M
    "ASN": 11,  # N
    "PRO": 12,  # P
    "GLN": 13,  # Q
    "ARG": 14,  # R
    "SER": 15,  # S
    "THR": 16,  # T
    "VAL": 17,  # V
    "TRP": 18,  # W
    "TYR": 19,  # Y
    # Common modified residues mapped to their parent amino acid
    "MSE": 10,  # selenomethionine  → MET
    "SEP": 15,  # phosphoserine     → SER
    "TPO": 16,  # phosphothreonine  → THR
    "PTR": 19,  # phosphotyrosine   → TYR
    "HSD": 6,   # CHARMM His (Nδ)  → HIS
    "HSE": 6,   # CHARMM His (Nε)  → HIS
    "HSP": 6,   # CHARMM His (both) → HIS
    "HIP": 6,   # AMBER His+        → HIS
    "HIE": 6,   # AMBER His (Nε)    → HIS
    "HID": 6,   # AMBER His (Nδ)    → HIS
    "CSS": 1,   # S-carbamidomethyl-Cys → CYS
    "CME": 1,   # S,S-dimethylarsinoyl  → CYS
}

_BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")
_WATER_NAMES = frozenset({"HOH", "WAT", "H2O", "DOD", "DIS", "D2O"})

# ── Element → atomic number (common small-molecule elements only) ─────────────
# The _encode_ligand_elements function bins these into 6 groups anyway;
# this map is for populating Y_t which expects raw atomic numbers.
_ELEM_TO_ATOMIC_NUM: Dict[str, int] = {
    "H": 1,  "D": 1,   "B": 5,   "C": 6,   "N": 7,   "O": 8,   "F": 9,
    "SI": 14, "P": 15,  "S": 16,  "CL": 17, "SE": 34, "BR": 35, "I": 53,
    "FE": 26, "ZN": 30, "MG": 12, "CA": 20, "MN": 25, "CU": 29,
    "CO": 27, "NI": 28, "NA": 11, "K": 19,
}


def parse_pdb(
    pdb_path: str,
    cutoff_for_score: float = 8.0,
    device: str = "cpu",
    parse_chains: Optional[List[str]] = None,
    include_zero_occupancy: bool = False,
) -> Dict[str, object]:
    """Parse a PDB file into tensors compatible with ``load_example_from_pdb``.

    Args:
        pdb_path: Path to a ``.pdb`` file.
        cutoff_for_score: Ligand atoms within this Å distance of any Cα are
            marked valid in ``Y_m``.  Distant atoms (crystal contacts, symmetry
            mates) are excluded.  Falls back to all atoms when no protein Cα
            is found.
        device: Torch device string.
        parse_chains: When provided, only residues whose chain ID is in this
            list are included. Ligand atoms are still parsed from every chain.
        include_zero_occupancy: When False (default) atoms with occupancy=0
            are skipped, matching BioPython's default PDBParser behaviour.
            Set True to mirror LigandMPNN's ``parse_atoms_with_zero_occupancy=1``.

    Returns:
        Dict with:

        * ``X``              ``[L, 4, 3]`` backbone coords (N, CA, C, O);
          rows with a missing Cα are all-zeros.
        * ``S``              ``[L]`` int64 sequence token (0-19 = AA, 20 = X).
        * ``mask``           ``[L]`` bool — True where Cα is present.
        * ``chain_mask``     ``[L]`` bool — True for all residues (all chains
          are treated as designable by default).
        * ``chain_ids``      ``List[str]`` length L — one-letter chain ID
          per residue, aligned with ``X`` / ``S`` / ``mask``.
        * ``res_nums``       ``List[int]`` length L — PDB residue number
          per residue.
        * ``insertion_codes`` ``List[str]`` length L — PDB insertion code
          per residue (empty string when absent).
        * ``Y``              ``[M, 3]`` float32 ligand heavy-atom coordinates.
        * ``Y_t``            ``[M]`` int64 atomic numbers.
        * ``Y_m``            ``[M]`` bool — True for atoms within
          *cutoff_for_score* Å of any Cα.

    Raises:
        ImportError: If BioPython is not installed.
        ValueError: If no protein residues are found in the file.
        FileNotFoundError: If *pdb_path* does not exist.
    """
    try:
        from Bio.PDB import PDBParser as _BPParser
    except ImportError as exc:
        raise ImportError(
            "biopython is required for PDB parsing. Install with: uv add biopython"
        ) from exc

    if not __import__("os").path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    bp_parser = _BPParser(QUIET=True)
    structure = bp_parser.get_structure("s", pdb_path)
    model = next(iter(structure))  # first MODEL record

    parse_chain_set = set(parse_chains) if parse_chains else None

    # Accumulate per-residue backbone data and ligand atoms
    backbone_rows: List[Tuple[torch.Tensor, int, bool]] = []  # (coords[4,3], token, ca_ok)
    residue_meta: List[Tuple[str, int, str]] = []  # (chain_id, res_num, insertion_code)
    ca_coords_list: List[torch.Tensor] = []                   # valid Cα positions
    ligand_atoms: List[Tuple[torch.Tensor, int]] = []         # (coord[3], atomic_num)

    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            het_flag, res_num, icode = residue.get_id()

            if het_flag == " ":
                # Standard amino acid (ATOM record)
                if parse_chain_set is not None and chain_id not in parse_chain_set:
                    continue

                resname = residue.get_resname().strip()
                token = _AA3_TO_TOKEN.get(resname, 20)

                coords = torch.zeros(4, 3, dtype=torch.float32)
                ca_ok = False
                for bb_idx, atom_name in enumerate(_BACKBONE_ATOMS):
                    if residue.has_id(atom_name):
                        atom = residue[atom_name]
                        if not include_zero_occupancy and atom.get_occupancy() == 0.0:
                            continue
                        # get_coord() returns the first alternate conformer
                        coords[bb_idx] = torch.tensor(
                            atom.get_coord(), dtype=torch.float32
                        )
                        if atom_name == "CA":
                            ca_ok = True
                            ca_coords_list.append(coords[1].clone())

                backbone_rows.append((coords, token, ca_ok))
                residue_meta.append((chain_id, int(res_num), icode.strip()))

            elif het_flag == "W":
                # Water — skip
                continue

            else:
                # HETATM ligand (het_flag = 'H_<resname>')
                resname = residue.get_resname().strip()
                if resname in _WATER_NAMES:
                    continue
                for atom in residue.get_atoms():
                    if not include_zero_occupancy and atom.get_occupancy() == 0.0:
                        continue
                    elem_raw = (atom.element or atom.get_name()[0]).strip().upper()
                    if elem_raw in ("H", "D"):
                        continue  # skip hydrogens / deuterium
                    # Unknown elements fall back to 119 (a sentinel past the
                    # periodic table). The v1 one-hot path routes 119 into its
                    # existing "other" bin; the v2 embedding path gets a
                    # dedicated learnable slot — previously these aliased to
                    # carbon, giving exotic elements the wrong signal.
                    atomic_num = _ELEM_TO_ATOMIC_NUM.get(elem_raw, 119)
                    coord = torch.tensor(atom.get_coord(), dtype=torch.float32)
                    ligand_atoms.append((coord, atomic_num))

    if not backbone_rows:
        raise ValueError(f"No protein residues found in {pdb_path}")

    L = len(backbone_rows)
    X = torch.stack([r[0] for r in backbone_rows])                           # [L, 4, 3]
    S = torch.tensor([r[1] for r in backbone_rows], dtype=torch.long)        # [L]
    mask = torch.tensor([r[2] for r in backbone_rows], dtype=torch.bool)     # [L]
    chain_mask = torch.ones(L, dtype=torch.bool)

    if ligand_atoms:
        Y = torch.stack([a[0] for a in ligand_atoms])                        # [M, 3]
        Y_t = torch.tensor([a[1] for a in ligand_atoms], dtype=torch.long)   # [M]
    else:
        Y = torch.zeros((0, 3), dtype=torch.float32)
        Y_t = torch.zeros(0, dtype=torch.long)

    # Mark ligand atoms within cutoff_for_score of any Cα
    if Y.shape[0] > 0 and ca_coords_list:
        ca_tensor = torch.stack(ca_coords_list)                              # [L_valid, 3]
        dists = torch.cdist(Y, ca_tensor)                                    # [M, L_valid]
        Y_m = dists.min(dim=1).values < cutoff_for_score                     # [M]
        if not Y_m.any():
            logger.debug(
                "%s: no ligand atoms within %.1f Å of Cα — keeping all %d atoms",
                pdb_path,
                cutoff_for_score,
                Y.shape[0],
            )
            Y_m = torch.ones(Y.shape[0], dtype=torch.bool)
    else:
        Y_m = torch.ones(Y.shape[0], dtype=torch.bool)

    chain_ids = [m[0] for m in residue_meta]
    res_nums = [m[1] for m in residue_meta]
    insertion_codes = [m[2] for m in residue_meta]

    return {
        "X": X.to(device),
        "S": S.to(device),
        "mask": mask.to(device),
        "chain_mask": chain_mask.to(device),
        "chain_ids": chain_ids,
        "res_nums": res_nums,
        "insertion_codes": insertion_codes,
        "Y": Y.to(device),
        "Y_t": Y_t.to(device),
        "Y_m": Y_m.to(device),
    }
