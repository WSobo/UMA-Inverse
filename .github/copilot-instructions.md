# GitHub Copilot Agent Instructions

## Environment & Execution Rules

**CRITICAL: Never run Python scripts directly with `python script.py` on this machine.**
This is an HPC cluster. All Python execution must go through SLURM or an interactive GPU allocation.

### Package manager: uv

This project uses **uv** (not conda/micromamba). All Python commands must be prefixed with `uv run`:

```bash
uv run python scripts/train.py
uv run python -m pytest tests/
uv run ruff check src/
```

Install/sync the environment with:
```bash
uv sync           # runtime deps only
uv sync --extra dev  # include pytest, ruff
```

### For quick tests / one-off commands:

Always wrap execution using `srun`:
```bash
srun --partition=gpu --gres=gpu:A5500:1 --mem=32G --cpus-per-task=8 --time=00:30:00 \
  bash -c "cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Inverse && uv run python <script>"
```

### For longer jobs:

Use `sbatch scripts/SLURM/<relevant_script>.sh`

### Never do:
- `python script.py` — no GPU, wrong environment
- `pip install ...` — use `uv add` or edit pyproject.toml
- `conda activate` / `micromamba activate` — use uv

### Do not touch these reference repos:
- `LigandMPNN/`
- `Struct2Seq-GNN/`
- `UMA-Fold/`
