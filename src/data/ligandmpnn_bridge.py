import importlib.util
import json
import os
from typing import Dict, List, Optional

import torch


def _load_ligandmpnn_data_utils():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    candidates = [
        os.path.join(project_root, "..", "LigandMPNN", "data_utils.py"),
        os.path.join(project_root, "LigandMPNN", "data_utils.py"),
    ]

    path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            "Could not find LigandMPNN/data_utils.py. Keep LigandMPNN at ../LigandMPNN."
        )

    spec = importlib.util.spec_from_file_location("ligandmpnn_data_utils", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load LigandMPNN data_utils.py loader.")
    spec.loader.exec_module(module)
    return module


_LIGANDMPNN = _load_ligandmpnn_data_utils()
PARSE_PDB = _LIGANDMPNN.parse_PDB
FEATURIZE = _LIGANDMPNN.featurize


def load_json_ids(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return [str(x) for x in ids]


def resolve_pdb_path(pdb_dir: str, pdb_id: str) -> Optional[str]:
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


def _compute_backbone_dihedrals(x: torch.Tensor) -> torch.Tensor:
    # x: [L, 4, 3] where atom order is N, CA, C, O
    n = x[:, 0, :]
    ca = x[:, 1, :]
    c = x[:, 2, :]

    c_prev = torch.cat([c[:1], c[:-1]], dim=0)
    n_next = torch.cat([n[1:], n[-1:]], dim=0)
    ca_next = torch.cat([ca[1:], ca[-1:]], dim=0)

    def _dihedral(p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        b1_norm = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + 1e-8)
        n1 = torch.linalg.cross(b0, b1_norm, dim=-1)
        n2 = torch.linalg.cross(b1_norm, b2, dim=-1)
        m1 = torch.linalg.cross(n1, b1_norm, dim=-1)

        x_val = (n1 * n2).sum(dim=-1)
        y_val = (m1 * n2).sum(dim=-1)
        return torch.atan2(y_val, x_val)

    phi = _dihedral(c_prev, n, ca, c)
    psi = _dihedral(n, ca, c, n_next)
    omega = _dihedral(ca, c, n_next, ca_next)

    angles = torch.stack([phi, psi, omega], dim=-1)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _encode_ligand_elements(y_t: torch.Tensor) -> torch.Tensor:
    # 6 bins: C, N, O, S, P, other
    out = torch.zeros((y_t.shape[0], 6), dtype=torch.float32, device=y_t.device)
    mask_c = y_t == 6
    mask_n = y_t == 7
    mask_o = y_t == 8
    mask_s = y_t == 16
    mask_p = y_t == 15
    out[mask_c, 0] = 1.0
    out[mask_n, 1] = 1.0
    out[mask_o, 2] = 1.0
    out[mask_s, 3] = 1.0
    out[mask_p, 4] = 1.0
    out[~(mask_c | mask_n | mask_o | mask_s | mask_p), 5] = 1.0
    return out


def _select_residue_crop(
    residue_coords: torch.Tensor,
    ligand_coords: torch.Tensor,
    max_residues: int,
) -> torch.Tensor:
    if residue_coords.shape[0] <= max_residues:
        return torch.arange(residue_coords.shape[0], device=residue_coords.device)

    if ligand_coords.shape[0] == 0:
        return torch.arange(max_residues, device=residue_coords.device)

    ligand_center = ligand_coords.mean(dim=0, keepdim=True)
    dist = torch.linalg.norm(residue_coords - ligand_center, dim=-1)
    topk = torch.topk(dist, k=max_residues, largest=False).indices
    topk, _ = torch.sort(topk)
    return topk


def load_example_from_pdb(
    pdb_path: str,
    ligand_context_atoms: int = 25,
    cutoff_for_score: float = 8.0,
    max_total_nodes: int = 384,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    protein_dict, *_ = PARSE_PDB(pdb_path, device=device)

    if "chain_mask" not in protein_dict:
        protein_dict["chain_mask"] = torch.ones_like(protein_dict["mask"], dtype=torch.int32)

    feat = FEATURIZE(
        protein_dict,
        cutoff_for_score=cutoff_for_score,
        use_atom_context=True,
        number_of_ligand_atoms=ligand_context_atoms,
        model_type="ligand_mpnn",
    )

    x = feat["X"][0]  # [L, 4, 3]
    residue_mask = feat["mask"][0].bool()
    residue_coords = x[:, 1, :][residue_mask]
    residue_features = _compute_backbone_dihedrals(x)[residue_mask]
    sequence = feat["S"][0][residue_mask].long()
    design_mask = feat["chain_mask"][0][residue_mask].bool()

    y = protein_dict["Y"]
    y_t = protein_dict["Y_t"]
    y_m = protein_dict["Y_m"].bool()
    ligand_coords = y[y_m]
    ligand_elements = y_t[y_m]

    if ligand_coords.numel() == 0:
        ligand_coords = torch.zeros((0, 3), dtype=torch.float32, device=residue_coords.device)
        ligand_features = torch.zeros((0, 6), dtype=torch.float32, device=residue_coords.device)
    else:
        # Keep nearest ligand atoms to the current residue cloud for fixed memory budget.
        if ligand_coords.shape[0] > ligand_context_atoms:
            residue_center = residue_coords.mean(dim=0, keepdim=True)
            dist = torch.linalg.norm(ligand_coords - residue_center, dim=-1)
            keep = torch.topk(dist, k=ligand_context_atoms, largest=False).indices
            ligand_coords = ligand_coords[keep]
            ligand_elements = ligand_elements[keep]
        ligand_features = _encode_ligand_elements(ligand_elements)

    max_residues = max(1, max_total_nodes - ligand_coords.shape[0])
    keep_idx = _select_residue_crop(
        residue_coords=residue_coords,
        ligand_coords=ligand_coords,
        max_residues=max_residues,
    )

    residue_coords = residue_coords[keep_idx]
    residue_features = residue_features[keep_idx]
    sequence = sequence[keep_idx]
    design_mask = design_mask[keep_idx]

    return {
        "residue_coords": residue_coords.float(),
        "residue_features": residue_features.float(),
        "residue_mask": torch.ones(residue_coords.shape[0], dtype=torch.bool),
        "sequence": sequence.long(),
        "design_mask": design_mask.bool(),
        "ligand_coords": ligand_coords.float(),
        "ligand_features": ligand_features.float(),
        "ligand_mask": torch.ones(ligand_coords.shape[0], dtype=torch.bool),
    }
