#!/bin/bash
#SBATCH --job-name=lmpnn-iface-designs
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/lmpnn_iface_designs_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/lmpnn_iface_designs_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Generate LigandMPNN designs on the three interface test splits for an
# apples-to-apples interface-recovery comparison vs UMA-Inverse:
#   - 10 designs/PDB, T=0.1, ligand_mpnn model (atom context ON)
#   - ALL chains designed (no --fixed_residues / no --chains_to_design) so the
#     recovery is scored over the same residues UMA designs.
# Scored afterwards by 05r via scripts/benchmark_interface_recovery.py
# --ligandmpnn-seqs-dir (identical interface mask + per-PDB-median aggregation).

set -e
PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "$PROJ"
LIGDIR=/private/groups/yehlab/wsobolew/01_software/LigandMPNN
CKPT="$LIGDIR/model_params/ligandmpnn_v_32_010_25.pt"
NUM="${NUM:-10}"

[[ -f "$CKPT" ]] || { echo "FATAL: LigandMPNN weights not found: $CKPT" >&2; exit 1; }

for cls in small_molecule metal nucleotide; do
    OUT="outputs/benchmark/ligandmpnn_interface/$cls"
    mkdir -p "$OUT/inputs/chunks"

    echo ">> [$cls] building input JSON  ($(date))"
    uv run python - "$cls" "$OUT" <<'PY'
import sys, json
from pathlib import Path
sys.path.insert(0, ".")
from src.data.ligandmpnn_bridge import load_json_ids, resolve_pdb_path
cls, out = sys.argv[1], Path(sys.argv[2])
ids = load_json_ids(f"LigandMPNN/training/test_{cls}.json")
# Absolute pdb_dir -> absolute paths in the chunk JSON. LigandMPNN run.py runs
# after `cd $LIGDIR`, so relative paths would resolve against the wrong dir.
pdb_dir = str(Path(f"data/raw/pdb_archive/test_{cls}").resolve())
paths, missing = {}, []
for pid in ids:
    p = resolve_pdb_path(pdb_dir, pid)
    if p:
        paths[str(p)] = ""
    else:
        missing.append(pid)
chunks_dir = out / "inputs" / "chunks"
plist = list(paths)
chunk_files = []
for i in range(0, len(plist), 100):
    cf = chunks_dir / f"chunk_{i//100:03d}.json"
    cf.write_text(json.dumps({p: "" for p in plist[i:i+100]}, indent=2))
    chunk_files.append(str(cf.resolve()))  # absolute: run.py reads this after cd $LIGDIR
(chunks_dir / "chunk_list.txt").write_text("\n".join(chunk_files) + "\n")
print(f"{cls}: resolved {len(paths)} PDBs ({len(missing)} missing) -> {len(chunk_files)} chunks")
PY

    echo ">> [$cls] running LigandMPNN ($NUM designs/PDB, all chains)  ($(date))"
    (
        eval "$(micromamba shell hook --shell bash)"
        micromamba activate ligandmpnn_env
        cd "$LIGDIR"
        while IFS= read -r chunk; do
            [[ -n "$chunk" ]] || continue
            python run.py \
                --model_type ligand_mpnn \
                --checkpoint_ligand_mpnn "$CKPT" \
                --pdb_path_multi "$chunk" \
                --out_folder "$PROJ/$OUT" \
                --batch_size 1 --number_of_batches "$NUM" \
                --temperature 0.1 --seed 0 --save_stats 1 \
                || echo "!! chunk $chunk failed -- continuing"
        done < "$PROJ/$OUT/inputs/chunks/chunk_list.txt"
    )
    nseq=$(ls "$OUT"/seqs/*.fa 2>/dev/null | wc -l)
    echo ">> [$cls] produced $nseq design FASTAs  ($(date))"
    [[ "$nseq" -gt 0 ]] || echo "!! WARNING: 0 designs produced for $cls"
done

total=0
for cls in small_molecule metal nucleotide; do
    total=$((total + $(ls "outputs/benchmark/ligandmpnn_interface/$cls"/seqs/*.fa 2>/dev/null | wc -l)))
done
echo ">> all splits done: $total total design FASTAs  ($(date))"
[[ "$total" -gt 0 ]] || { echo "FATAL: no designs produced across any split" >&2; exit 1; }
