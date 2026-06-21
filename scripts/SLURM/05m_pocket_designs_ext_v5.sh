#!/bin/bash
#SBATCH --job-name=pocket-designs-ext-v5
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_out/pocket_designs_ext_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse/logs/SLURM_err/pocket_designs_ext_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Stage 1 of the cofold scale-up: generate pocket-fixed designs for the 69 NEW
# small-molecule PDBs (outputs/preprint/pdb_selection_extended_v5.json). UMA and
# LigandMPNN designs are written alongside the existing 35-set designs; the new
# PDB ids are disjoint from the existing ones, so nothing is overwritten and the
# already-cofolded 35 stay reusable. Same ckpt/config/T/seed as the original 35.

set -e
PROJ=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse
cd "$PROJ"
EXT=outputs/preprint/pdb_selection_extended_v5.json

# ── [1/2] UMA pocket-fixed designs (uv venv) ────────────────────────────────
echo ">> [1/2] UMA pocket-fixed designs (extended, 69 new)  ($(date))"
uv run python scripts/paper/run_pocket_fixed_designs.py \
    --ckpt checkpoints/pairmixerinv-v5-stage3-nodes384-ddp2/epoch_snapshots/epoch-11.ckpt \
    --config configs/config.yaml \
    --selection "$EXT" \
    --out-dir outputs/preprint/uma_pocket_fixed_v5 \
    --num-samples 20 --temperature 0.1 --seed 0

# ── [2/2] LigandMPNN pocket-fixed designs (ligandmpnn_env) ───────────────────
echo ">> [2/2] LigandMPNN pocket-fixed designs (extended, 69 new)  ($(date))"
uv run python scripts/paper/build_ligandmpnn_inputs.py \
    --selection "$EXT" \
    --out-dir outputs/preprint/ligandmpnn_inputs_ext_v5
(
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate ligandmpnn_env
    LIGDIR=/private/groups/yehlab/wsobolew/01_software/LigandMPNN
    CKPT="$LIGDIR/model_params/ligandmpnn_v_32_010_25.pt"
    cd "$LIGDIR"
    python run.py \
        --model_type ligand_mpnn \
        --checkpoint_ligand_mpnn "$CKPT" \
        --pdb_path_multi "$PROJ/outputs/preprint/ligandmpnn_inputs_ext_v5/pdb_path_multi.json" \
        --fixed_residues_multi "$PROJ/outputs/preprint/ligandmpnn_inputs_ext_v5/fixed_residues_multi.json" \
        --chains_to_design A \
        --out_folder "$PROJ/outputs/preprint/ligandmpnn_pocket_fixed" \
        --batch_size 1 --number_of_batches 20 \
        --temperature 0.1 --seed 0 --save_stats 1
)
echo ">> Stage 1 (designs) complete  ($(date))"
