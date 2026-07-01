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
configs/          # Hydra config surface (config.yaml = the v5 architecture)
  config.yaml     # Main config: paths, training, data, model, wandb
  data/ model/ trainer/   # Grouped hyperparams
  old_configs/    # Archived earlier configs (v1, v3) for reference
src/
  data/           # LigandMPNN bridge, DataModule, vendored PDB parser
  models/         # RBFEmbedding, PairMixerBlock, UMAInverse model
  training/       # UMAInverseLightningModule + distogram aux head
  inference/      # InferenceSession, decoding, constraints, weights; CLI (`uma-inverse`)
  benchmarks/     # Interface-recovery evaluation + metrics
  serving/        # FastAPI REST + Gradio UI + Prometheus/structlog (CPU service)
  mcp/            # MCP server: design/score as agent tools (talks to the REST API)
  utils/          # FASTA / token I/O helpers
scripts/
  train.py · pilot_run.py · preprocess.py · preprocess_v5.py  # train + data prep
  download_json_pdbs.py · download_weights.py                 # fetch PDBs / HF weights
  precompute_examples.py                                      # cache serving demo results
  benchmark_interface_recovery.py · …                         # paper metrics
  SLURM/          # sbatch wrappers for each stage
  paper/          # preprint figure + table generation
tests/            # pytest unit/integration (CPU-only); incl. tests/test_serving/
Dockerfile · deploy/hf_space/   # CPU serving image + HF Spaces (Docker SDK) deploy
.github/workflows/ci.yml        # CI: ruff + pytest + docker-build (all CPU)
```

## Workflow (Order of Operations)
```
01a  make download      # sbatch: fetch PDB structures from RCSB
01b  make preprocess    # sbatch: cache PDBs as .pt tensors in data/processed/
02   make pilot         # srun:   1-batch overfit sanity check
03   make train         # sbatch: 3-stage curriculum training (nodes 64→128→384)
04   make inference PDB=path/to/file.pdb   # srun: design sequences → FASTA
                        #   (or: uv run uma-inverse design --pdb ...)
05   make serve         # CPU REST + Gradio service on :7860 (no srun); `make mcp` = agent tool
```

## Key Config Parameters
All Hydra overrides use `++key=value` syntax on the CLI. `configs/config.yaml` is
now the **v5 architecture** — any served/inferred checkpoint must match it.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `data.max_total_nodes` | 384 | Curriculum stages: 64 → 128 → 384 |
| `data.ligand_context_atoms` | 50 | Max ligand atoms kept (v5 raised 25 → 50 for DNA/RNA) |
| `data.ligand_featurizer` | ligandmpnn_atomic | v5 atomic featurizer (`ligand_in.proj`) |
| `data.frame_relative_angles` | true | v5 per-(residue, ligand) bearing angles |
| `data.pair_distance_atoms` | backbone_full_25 | v5 multi-atom residue–residue distances |
| `training.lr` | 3e-4 | AdamW learning rate |
| `model.num_pairmixer_blocks` | 6 | Encoder depth |
| `model.pair_dim` | 128 | Pair tensor channel dimension |
| `wandb.enabled` | false | Set true to enable W&B logging |

## Packaging & Dev Tools
- **Package manager:** uv (`uv sync`, `uv run`, `uv add`)
- **Linter:** ruff (`make lint` / `make lint-fix`)
- **Tests:** pytest (`make test` / `make test-fast`)
- **Build:** hatchling (wheels from `src/` only)
- **Dev extras:** `uv sync --extra dev`
- **Serving:** `uv sync --extra serving`; `make serve` (REST+UI), `make mcp` (agent tool),
  `make docker-build` (CPU image). Weights auto-fetch from HF (`WSobo/UMA-Inverse`); deploy
  steps in `deploy/hf_space/`. Serving runs OFF the SLURM cluster — plain commands, no `srun`.

