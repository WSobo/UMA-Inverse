"""Tests for src.data.pdb_parser, focused on the v5 DNA/RNA routing change.

Nucleotide ATOM records (chain residues with names like DA/DC/DG/DT/A/U)
were silently dropped pre-v5 because they appear with het_flag=' ' and
LigandMPNN's parser only knew about standard amino-acid ATOM records. v5
routes them into the ligand atom pool so protein+nucleic-acid complexes
can be featurised end-to-end.
"""
from pathlib import Path

import torch

from src.data.pdb_parser import parse_pdb


def _atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float, y: float, z: float,
    element: str,
) -> str:
    """Format one ATOM record per the PDB v3.3 spec (cols 1-80).

    Field widths matter: BioPython's parser slices by column, so even a
    single off-by-one in the atom-name or resname field misaligns the
    chain ID and resseq downstream.
    """
    # Atom name: 4-char field at cols 13-16. PDB convention is that
    # one-character element names (C, N, O, P, S) are placed in col 14
    # with cols 13, 15-16 blank; multi-character names like "C1'" right-
    # extend. Easiest robust rule: right-justify atom-name to width 4.
    name_field = f"{name:>4s}"[:4]
    resname_field = f"{resname:>3s}"[:3]
    element_field = f"{element:>2s}"[:2]
    return (
        f"ATOM  "                 # cols 1-6
        f"{serial:>5d}"           # cols 7-11
        f" "                      # col 12
        f"{name_field}"           # cols 13-16
        f" "                      # col 17 (alt-loc)
        f"{resname_field}"        # cols 18-20
        f" "                      # col 21
        f"{chain:>1s}"            # col 22
        f"{resseq:>4d}"           # cols 23-26
        f" "                      # col 27 (insertion)
        f"   "                    # cols 28-30
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"  # cols 31-54
        f"{1.00:>6.2f}{20.00:>6.2f}"    # cols 55-66 occ + temp
        f"          "             # cols 67-76
        f"{element_field}"        # cols 77-78
        f"  "                     # cols 79-80
    )


def _write_pdb(tmp_path: Path, lines: list[str]) -> str:
    """Write minimal PDB content to a temp file and return the path."""
    path = tmp_path / "synth.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return str(path)


# One ALA residue (N/CA/C/O) so the parser doesn't raise "no protein residues".
_BASE_PROTEIN_LINES = [
    _atom_line(1, "N",  "ALA", "A", 1, 11.104, 13.207, 2.500, "N"),
    _atom_line(2, "CA", "ALA", "A", 1, 11.804, 14.000, 3.000, "C"),
    _atom_line(3, "C",  "ALA", "A", 1, 13.000, 14.500, 2.500, "C"),
    _atom_line(4, "O",  "ALA", "A", 1, 13.500, 15.500, 3.000, "O"),
]


def test_dna_residues_route_into_ligand_pool(tmp_path):
    """DA/DC ATOM records must show up in Y (ligand coords), not as protein
    residues. Counts: 1 protein residue, 5 DNA heavy atoms (P, C1', N9 in
    DA + P, C1' in DC).
    """
    dna_lines = [
        _atom_line(5, "P",   "DA", "B", 1, 20.000,  5.000,  8.000, "P"),
        _atom_line(6, "C1'", "DA", "B", 1, 21.000,  6.000,  9.000, "C"),
        _atom_line(7, "N9",  "DA", "B", 1, 22.000,  7.000, 10.000, "N"),
        _atom_line(8, "P",   "DC", "B", 2, 23.000,  8.000, 11.000, "P"),
        _atom_line(9, "C1'", "DC", "B", 2, 24.000,  9.000, 12.000, "C"),
    ]
    pdb_path = _write_pdb(tmp_path, _BASE_PROTEIN_LINES + dna_lines)
    parsed = parse_pdb(pdb_path, cutoff_for_score=100.0)

    # Exactly one protein residue (ALA).
    assert int(parsed["mask"].sum().item()) == 1
    assert parsed["S"].shape == (1,)

    # All five DNA heavy atoms route into Y; none into the protein backbone.
    assert parsed["Y"].shape[0] == 5
    assert parsed["Y_t"].dtype == torch.long
    # P=15, C=6, N=7 — every DNA atom should map via _ELEM_TO_ATOMIC_NUM.
    assert set(parsed["Y_t"].tolist()) <= {6, 7, 15}


def test_rna_and_modified_nucleotides_routed(tmp_path):
    """Plain RNA (A, U) and a modified variant (PSU) must also be routed."""
    rna_lines = [
        _atom_line(5, "P",  "A",   "B", 1, 20.000, 5.000,  8.000, "P"),
        _atom_line(6, "N1", "U",   "B", 2, 21.000, 6.000,  9.000, "N"),
        _atom_line(7, "C2", "PSU", "B", 3, 22.000, 7.000, 10.000, "C"),
    ]
    pdb_path = _write_pdb(tmp_path, _BASE_PROTEIN_LINES + rna_lines)
    parsed = parse_pdb(pdb_path, cutoff_for_score=100.0)
    assert int(parsed["mask"].sum().item()) == 1
    assert parsed["Y"].shape[0] == 3


def test_protein_only_pdb_has_empty_ligand_tensor(tmp_path):
    """Regression: a no-DNA file should still parse and emit an empty Y."""
    pdb_path = _write_pdb(tmp_path, _BASE_PROTEIN_LINES)
    parsed = parse_pdb(pdb_path, cutoff_for_score=100.0)
    assert int(parsed["mask"].sum().item()) == 1
    assert parsed["Y"].shape == (0, 3)
    assert parsed["Y_t"].shape == (0,)


def test_dna_hydrogen_atoms_are_skipped(tmp_path):
    """Hydrogen / deuterium nucleotide atoms must be excluded from Y."""
    dna_lines = [
        _atom_line(5, "P",   "DA", "B", 1, 20.000, 5.000,  8.000, "P"),
        _atom_line(6, "H1'", "DA", "B", 1, 21.000, 6.000,  9.000, "H"),
        _atom_line(7, "H2'", "DA", "B", 1, 22.000, 7.000, 10.000, "D"),
    ]
    pdb_path = _write_pdb(tmp_path, _BASE_PROTEIN_LINES + dna_lines)
    parsed = parse_pdb(pdb_path, cutoff_for_score=100.0)
    # Only the P atom should be kept.
    assert parsed["Y"].shape[0] == 1
    assert int(parsed["Y_t"][0].item()) == 15  # P
