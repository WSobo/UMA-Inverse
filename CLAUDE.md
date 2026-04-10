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
  bash -c "cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse && uv run python scripts/train.py"

# ✅ Correct (batch job)
sbatch scripts/SLURM/03_train_model.sh

# ✅ Correct (CPU tests via Makefile)
make test
```

## Do Not Touch
These directories are **reference implementations only** — never read or modify:
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

---

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (90-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk vitest run          # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%)
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->
