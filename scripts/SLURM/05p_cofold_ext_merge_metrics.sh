#!/bin/bash
#SBATCH --job-name=cofold-ext-merge
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/cofold_ext_merge_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/cofold_ext_merge_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=medium

# Stage 4: merge the per-chunk Boltz outputs into the canonical cofold/ tree,
# compute metrics over the full 104-PDB set, and print the paired statistics.
# CPU-only. Runs afterany the array so it summarizes whatever completed.

set -e
PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "$PROJ"
COMB=outputs/preprint/pdb_selection_combined_v5.json

echo ">> [1/3] merging per-chunk cofold outputs into canonical tree  ($(date))"
uv run python - <<'PY'
from pathlib import Path
import shutil
cof = Path("outputs/preprint/cofold")
chunks_out = Path("outputs/preprint/cofold_chunks_out")
methods = ["uma_v5", "native", "ligandmpnn"]
moved = 0
if chunks_out.exists():
    for ch in sorted(chunks_out.glob("chunk_*")):
        for m in methods:
            src = ch / m / f"boltz_results_{m}" / "predictions"
            if not src.exists():
                continue
            dst = cof / m / f"boltz_results_{m}" / "predictions"
            dst.mkdir(parents=True, exist_ok=True)
            for rec in src.iterdir():
                target = dst / rec.name
                if target.exists():
                    continue
                shutil.move(str(rec), str(target))
                moved += 1
print(f"merged {moved} new prediction records into {cof}")
PY

echo ">> [2/3] cofold_metrics over combined_v5 (104 PDBs)  ($(date))"
uv run python scripts/paper/cofold_metrics.py \
    --sampling-record outputs/preprint/boltz_inputs/cofold/sampling_record.json \
    --cofold-base outputs/preprint/cofold \
    --selection "$COMB" \
    --out outputs/preprint/cofold_metrics_v5_big.csv

echo ">> [3/3] paired statistics (UMA vs LigandMPNN, vs native floor)  ($(date))"
uv run python - <<'PY'
import csv, math, statistics as st
from collections import defaultdict
import random
random.seed(0)
try:
    from scipy.stats import wilcoxon
    HAVE=True
except Exception:
    HAVE=False
rows=list(csv.DictReader(open("outputs/preprint/cofold_metrics_v5_big.csv")))
def fv(x):
    try:
        v=float(x); return None if math.isnan(v) else v
    except: return None
def per_pdb(metric, kind=None):
    d=defaultdict(lambda: defaultdict(list))
    for r in rows:
        if kind and r["kind"]!=kind: continue
        v=fv(r[metric])
        if v is not None: d[r["pdb_id"]][r["method"]].append(v)
    return {p:{m:st.median(vs) for m,vs in mm.items() if vs} for p,mm in d.items()}
def paired(metric,a="uma_v5",b="ligandmpnn",kind=None):
    pp=per_pdb(metric,kind); A=[];B=[]
    for p,mm in pp.items():
        if a in mm and b in mm: A.append(mm[a]);B.append(mm[b])
    return A,B
def boot(diffs,n=10000):
    if not diffs: return (float('nan'),float('nan'))
    ms=sorted(st.median([random.choice(diffs) for _ in diffs]) for _ in range(n))
    return ms[int(.025*n)],ms[int(.975*n)]
print(f"total rows: {len(rows)}")
for label,kind in [("ALL",None),("small_molecule","small_molecule"),("metal","metal")]:
    A,B=paired("ligand_rmsd_best",kind=kind); d=[a-b for a,b in zip(A,B)]
    print(f"\n[ligand_rmsd_best | {label}] paired PDBs n={len(d)}")
    if not d:
        print("  (none)"); continue
    lo,hi=boot(d)
    p=wilcoxon(A,B).pvalue if (HAVE and len(d)>=6) else float('nan')
    print(f"  median UMA={st.median(A):.3f}  LigMPNN={st.median(B):.3f}  diff={st.median(d):+.3f} A")
    print(f"  UMA better on {sum(1 for x in d if x<0)}/{len(d)}   95%CI(diff)=[{lo:+.3f},{hi:+.3f}]   Wilcoxon p={p:.3f}")
for metric in ["pocket_calpha_rmsd_best","scaffold_rmsd_best","ligand_iptm_best"]:
    A,B=paired(metric); d=[a-b for a,b in zip(A,B)]
    p=wilcoxon(A,B).pvalue if (HAVE and len(d)>=6) else float('nan')
    print(f"[{metric}] n={len(d)} medUMA={st.median(A):.3f} medLig={st.median(B):.3f} diff={st.median(d):+.3f} p={p:.3f}")
A,B=paired("ligand_rmsd_best",a="uma_v5",b="native"); d=[a-b for a,b in zip(A,B)]
if d: print(f"\n[UMA vs NATIVE floor] n={len(d)}  medUMA={st.median(A):.3f} medNative={st.median(B):.3f} diff={st.median(d):+.3f}")
PY
echo ">> Stage 4 (merge+metrics+stats) complete  ($(date))"
