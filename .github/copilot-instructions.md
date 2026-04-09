# GitHub Copilot Agent Instructions

## Environment & Execution Rules

**CRITICAL: Never run Python scripts directly with `python script.py` on this machine.**
This is an HPC cluster. All Python execution must go through SLURM or an interactive GPU allocation.

### For quick tests / one-off commands:
Always wrap execution using `srun`:
```bash
srun --partition=gpu --gres=gpu:A5500:1 --mem=32G --cpus-per-task=8 --time=00:30:00 \
  bash -c "eval \"\$(micromamba shell hook --shell bash)\" && micromamba activate uma-fold && python "
```

### For longer jobs:
Use `sbatch scripts/SLURM/<relevant_script>.sh`

### Environment activation:
- Conda manager: **micromamba** (not conda, not mamba)
- Environment name: **uma-fold**
- Always activate before any Python/pip command:
```bash
  eval "$(micromamba shell hook --shell bash)" && micromamba activate uma-fold
```

### Never do:
- `python test.py` — no GPU, wrong environment
- `pip install ...` — use the uma-fold env
- `conda activate` — use micromamba