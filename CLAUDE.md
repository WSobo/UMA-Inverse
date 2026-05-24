# CLAUDE.md — UMA-Inverse

## Project
UMA-Inverse is a ligand-conditioned protein inverse-folding model using a dense invariant PairMixer encoder. Given fixed protein-ligand backbone coordinates, it predicts per-residue amino acid identity (21 tokens: 20 AA + X). It reuses LigandMPNN's parser/featurizer and targets single-GPU training (24 GB, bf16-mixed).

## CRITICAL: HPC Execution Rules

**Never run `python script.py` directly.** This is a SLURM HPC cluster.

- **Interactive (quick tests/one-offs):** wrap with `srun` — see Makefile targets
- **Long jobs:** `sbatch scripts/SLURM/<script>.sh`
- **Environment:** uv — always prefix Python commands with `uv run`
- **Never:** `pip install`, `conda activate`, `micromamba activate`, bare `python`

```bash
# ✅ Correct (interactive GPU)
srun --partition=gpu --gres=gpu:A5500:1 --mem=32G --cpus-per-task=8 --time=00:30:00 \
  bash -c "cd /path/to/UMA-Inverse && uv run python scripts/train.py"

# ✅ Correct (batch job)
sbatch scripts/SLURM/03_train_model.sh

# ✅ Correct (CPU tests via Makefile)
make test
```

## Do Not Touch
These directories are **reference implementations only** — never modify:
- `LigandMPNN/` — only its `training/train.json` and `training/valid.json` split files are used at runtime
- `Struct2Seq-GNN/`
- `UMA-Fold/`

The PDB parsing code (`parse_PDB`, `featurize`) has been vendorized into `src/data/pdb_parser.py` — **no code from `LigandMPNN/` is imported at runtime**.

## Repository Layout
```
configs/          # Hydra config surface (config.yaml is the root)
  config.yaml     # Main config: paths, training, data, model, wandb
  data/           # Data hyperparams (batch_size, max_total_nodes, etc.)
  model/          # PairMixer architecture hyperparams
  trainer/        # PyTorch Lightning trainer settings
src/
  data/           # LigandMPNN bridge + PyTorch Lightning DataModule
  models/         # RBFEmbedding, PairMixerBlock, UMAInverse model
  training/       # UMAInverseLightningModule (train/val step, optimizer)
  utils/          # FASTA I/O helpers
scripts/
  train.py        # Hydra training entry point
  inference.py    # Inference CLI (--pdb, --ckpt, --out_fasta)
  pilot_run.py    # Single-batch overfit sanity check
  preprocess.py   # Batch PDB → .pt tensor cache
  download_json_pdbs.py  # Fetch PDBs from RCSB
  SLURM/          # sbatch wrappers for each stage
tests/            # pytest unit/integration tests (CPU-only)
```

## Workflow (Order of Operations)
```
01a  make download      # sbatch: fetch PDB structures from RCSB
01b  make preprocess    # sbatch: cache PDBs as .pt tensors in data/processed/
02   make pilot         # srun:   1-batch overfit sanity check
03   make train         # sbatch: 3-stage curriculum training (nodes 64→128→384)
04   make inference PDB=path/to/file.pdb   # srun: design sequences → FASTA
```

## Key Config Parameters
All Hydra overrides use `++key=value` syntax on the CLI.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `data.max_total_nodes` | 384 | Curriculum stages: 64 → 128 → 384 |
| `data.ligand_context_atoms` | 25 | Max ligand atoms kept |
| `training.lr` | 3e-4 | AdamW learning rate |
| `training.epochs` | 10 | Overridden per curriculum stage |
| `model.num_pairmixer_blocks` | 6 | Encoder depth |
| `model.pair_dim` | 128 | Pair tensor channel dimension |
| `wandb.enabled` | false | Set true to enable W&B logging |

## Packaging & Dev Tools
- **Package manager:** uv (`uv sync`, `uv run`, `uv add`)
- **Linter:** ruff (`make lint` / `make lint-fix`)
- **Tests:** pytest (`make test` / `make test-fast`)
- **Build:** hatchling (wheels from `src/` only)
- **Dev extras:** `uv sync --extra dev`

