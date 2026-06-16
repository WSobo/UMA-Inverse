"""How much of the nucleic acid does UMA-Inverse's global atom budget capture?

For each nucleotide test PDB we replicate the model's ligand-atom selection (the
<=50 nucleic-acid heavy atoms nearest the protein CA centroid; see the topk in
src/data/ligandmpnn_bridge.py) and report:
  - total nucleic-acid heavy atoms in the complex,
  - the fraction retained by the fixed 50-atom budget,
  - the fraction of the true protein-NA interface (NA heavy atoms within 5 A of
    any protein heavy atom) that the kept-50 actually covers.

This is a descriptive statistic (no model inference) supporting the Discussion's
claim that the global centroid budget captures only a thin slice of a macromolecule.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
PDB_DIR = ROOT / "data" / "raw" / "pdb_archive" / "test_nucleotide"
BUDGET = 50
IFACE = 5.0

AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"}
NA = {"DA", "DC", "DG", "DT", "DU", "DI", "A", "C", "G", "U", "I"}


def parse(pdb: Path):
    ca, prot, na = [], [], []
    seen_model = False
    for line in pdb.read_text().splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            if seen_model:
                break          # first model only
            seen_model = True
            continue
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM", "HETATM"):
            continue
        if line[16] not in (" ", "A"):  # altLoc
            continue
        resn = line[17:20].strip()
        name = line[12:16].strip()
        elem = (line[76:78].strip() or name[:1]).upper()
        if elem == "H":
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if resn in AA:
            prot.append(xyz)
            if name == "CA":
                ca.append(xyz)
        elif resn in NA:
            na.append(xyz)
    return np.array(ca), np.array(prot), np.array(na)


rows = []
for pdb in sorted(PDB_DIR.rglob("*.pdb")):
    ca, prot, na = parse(pdb)
    if len(na) == 0 or len(ca) == 0 or len(prot) == 0:
        continue
    centroid = ca.mean(0, keepdims=True)
    d = np.linalg.norm(na - centroid, axis=1)
    keep = set(np.argsort(d)[:BUDGET].tolist())
    iface = cKDTree(prot).query_ball_point(na, IFACE)
    iface_idx = [i for i, hit in enumerate(iface) if hit]
    n_iface = len(iface_idx)
    cov = (sum(i in keep for i in iface_idx) / n_iface) if n_iface else float("nan")
    rows.append((pdb.stem, len(na), min(BUDGET, len(na)) / len(na), n_iface, cov))

na_counts = [r[1] for r in rows]
retained = [r[2] for r in rows]
covs = [r[4] for r in rows if r[4] == r[4]]
big = [r for r in rows if r[1] > BUDGET]

print(f"parsed {len(rows)} nucleotide structures ({len(big)} have >{BUDGET} NA heavy atoms)")
print(f"NA heavy atoms:           median {int(st.median(na_counts))}  "
      f"(IQR {int(np.percentile(na_counts,25))}-{int(np.percentile(na_counts,75))}, "
      f"range {min(na_counts)}-{max(na_counts)})")
print(f"fraction kept by budget:  median {st.median(retained)*100:.0f}%")
print(f"interface NA atoms (<{IFACE:.0f}A): median {int(st.median([r[3] for r in rows]))}")
print(f"interface covered by kept-50: median {st.median(covs)*100:.0f}%  mean {st.mean(covs)*100:.0f}%")
if big:
    print(f"  among the {len(big)} with >{BUDGET} NA atoms: "
          f"median kept {st.median([r[2] for r in big])*100:.0f}%, "
          f"median interface coverage {st.median([r[4] for r in big if r[4]==r[4]])*100:.0f}%")
