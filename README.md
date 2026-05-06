# UMA-Inverse

UMA-Inverse is a LigandMPNN-style inverse-folding project that swaps sparse message passing for a dense invariant PairMixer encoder.

## Why this architecture

- Comparable data protocol: uses LigandMPNN parsing and split JSON files.
- Dense geometry reasoning: replaces KNN-only neighborhood updates with triangle multiplication over pair features.
- Single-GPU friendly path: bf16 mixed precision, gradient checkpointing, and curriculum neighborhood cropping controls.
- Invariant-first design: uses distances and angle-derived local features, avoiding SE(3)-heavy kernels for v1.

## Model sketch

1. Parse protein-ligand structure with LigandMPNN `parse_PDB` and `featurize`.
2. Build node features:
   - Protein residues: backbone dihedral sin/cos + chain id embedding
   - Ligand atoms: element one-hot embedding
3. Build dense pair tensor `Z_ij` from:
   - node_i projection
   - node_j projection
   - RBF(distance(i, j))
4. Run stacked PairMixer blocks:
   - triangle multiplication outgoing
   - triangle multiplication incoming
   - transition MLP
5. Decode residue logits (20 AA + X) with ligand-aware context.

## Repository structure

```text
UMA-Inverse/
├── configs/
├── data/
├── logs/
├── scripts/
├── src/
└── tests/
```

## Quickstart

Please refer to [ORDER_OF_OPERATIONS.md](ORDER_OF_OPERATIONS.md) for the exact run sequence.

```bash
cd UMA-Inverse

# 1. Fetch & Preprocess
sbatch scripts/SLURM/01a_fetch_data.sh
sbatch scripts/SLURM/01b_preprocess.sh

# 2. Pilot / Overfit Test
sbatch scripts/SLURM/02_pilot_run.sh

# 3. Curriculum Training
sbatch scripts/SLURM/03_train_model.sh

# 4. Inference (design sequences)
uv run uma-inverse design --pdb my.pdb --ckpt checkpoints/last.ckpt --num-samples 10
```

## Inference

The `uma-inverse` CLI exposes three subcommands. See
[docs/inference.md](docs/inference.md) and [docs/benchmarks.md](docs/benchmarks.md)
for the full references.

```bash
# Design new sequences for a PDB
uv run uma-inverse design --pdb my.pdb --ckpt checkpoints/last.ckpt \
    --num-samples 10 --temperature 0.1 --top-p 0.95 --seed 42 \
    --fix "A1 A2 A3" --bias "W:3.0"

# Batch mode with a JSON spec and crash recovery
uv run uma-inverse design --pdb-list spec.json --ckpt ckpt.ckpt \
    --out-dir outputs/screen --num-samples 5 --resume

# Score native or user sequence
uv run uma-inverse score --pdb my.pdb --ckpt checkpoints/last.ckpt \
    --mode autoregressive --num-batches 10

# Full benchmark suite for a trained checkpoint (writes paper-ready tables + figures)
uv run uma-inverse benchmark --ckpt checkpoints/uma-inverse-best.ckpt \
    --val-json LigandMPNN/training/valid.json --pdb-dir data/raw/pdb_archive \
    --n-pdbs 500
```

## Single GPU defaults

- `precision: bf16-mixed`
- `batch_size: 1`
- `max_total_nodes: 384`
- `gradient_checkpointing: true`

## Notes

- This is an educational implementation for architecture experimentation.
- v1 focuses on sequence prediction quality and training stability.
- Sidechain coordinate prediction can be added as a v2 auxiliary head.
