# ==============================================================================
# UMA-Inverse Makefile
# ==============================================================================
# All GPU commands run through SLURM srun or sbatch automatically.
# Never run `python script.py` directly — this is an HPC cluster.
# Environment: uv (see pyproject.toml). All Python runs via `uv run python`.
#
# Usage:
#   make help         → list all targets
#   make pilot        → run 1-batch sanity check on GPU
#   make train        → submit full curriculum training job
#   make download     → submit dataset download job
#   make inference    → run inference on a PDB (set PDB=path/to/file.pdb)
#   make test         → run pytest suite (CPU, no GPU needed)
#   make lint         → run ruff linter
# ==============================================================================

# ── SLURM: Interactive GPU allocation (for srun one-off commands) ──────────────
SRUN_GPU := srun \
	--partition=gpu \
	--gres=gpu:A5500:1 \
	--mem=32G \
	--cpus-per-task=8 \
	--time=01:00:00

# ── SLURM: A100 interactive allocation (v4 training target) ────────────────────
SRUN_A100 := srun \
	--partition=gpu \
	--gres=gpu:A100:1 \
	--mem=40G \
	--cpus-per-task=8 \
	--time=01:00:00

# ── SLURM: CPU-only allocation (for lightweight jobs like pytest/lint) ─────────
SRUN_CPU := srun \
	--partition=medium \
	--mem=8G \
	--cpus-per-task=4 \
	--time=00:30:00

# ── Project paths ──────────────────────────────────────────────────────────────
CONFIG      := configs/config.yaml
CKPT        := checkpoints/last.ckpt


# ==============================================================================
# HELP (default target)
# ==============================================================================
.PHONY: help
help:
	@echo ""
	@echo "  UMA-Inverse Makefile"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make download       Submit SLURM dataset download job"
	@echo "  make preprocess     Submit SLURM preprocessing job"
	@echo "  make pilot          Run 1-batch sanity check on GPU (srun)"
	@echo "  make pilot-64       Pilot with max_total_nodes=64"
	@echo "  make pilot-128      Pilot with max_total_nodes=128"
	@echo "  make pilot-384      Pilot with max_total_nodes=384"
	@echo "  make pilot-all      Run all pilot stages via sbatch"
	@echo "  make pilot-v5       v5 1-batch sanity check (rich + bond + distogram, srun A100)"
	@echo "  make train          Submit full curriculum training (sbatch)"
	@echo "  make train-v5       Chain v5 3-stage curriculum via sbatch afterok"
	@echo "  make inference      Run inference — set PDB=path/to/file.pdb (srun)"
	@echo "  make test           Run pytest suite (srun, CPU)"
	@echo "  make test-fast      Run pytest, stop on first failure (srun, CPU)"
	@echo "  make lint           Run ruff linter (srun, CPU)"
	@echo "  make lint-fix       Run ruff linter with auto-fix (srun, CPU)"
	@echo "  make serve          Run the CPU REST + Gradio service locally (:7860, no srun)"
	@echo "  make mcp            Run the MCP agent server (talks to the REST API)"
	@echo "  make docker-build   Build the CPU serving image"
	@echo "  make precompute-examples  Cache bundled-example results for a snappy UI"
	@echo "  make env-check      Verify uv env + torch/CUDA (srun GPU)"
	@echo "  make jobs           List your queued/running SLURM jobs"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo ""


# ==============================================================================
# DATA
# ==============================================================================
.PHONY: download
download:
	sbatch scripts/SLURM/01a_fetch_data.sh

.PHONY: preprocess
preprocess:
	sbatch scripts/SLURM/01b_preprocess.sh


# ==============================================================================
# PILOT RUN (sanity check — 1 batch, fast_dev_run=True)
# ==============================================================================
.PHONY: pilot
pilot:
	$(SRUN_GPU) bash -c 'cd $(CURDIR) && uv run python scripts/pilot_run.py'

.PHONY: pilot-64
pilot-64:
	$(SRUN_GPU) bash -c 'cd $(CURDIR) && uv run python scripts/pilot_run.py ++data.max_total_nodes=64'

.PHONY: pilot-128
pilot-128:
	$(SRUN_GPU) bash -c 'cd $(CURDIR) && uv run python scripts/pilot_run.py ++data.max_total_nodes=128'

.PHONY: pilot-384
pilot-384:
	$(SRUN_GPU) bash -c 'cd $(CURDIR) && uv run python scripts/pilot_run.py ++data.max_total_nodes=384'

# Run all three pilot stages back-to-back in one sbatch allocation
.PHONY: pilot-all
pilot-all:
	sbatch scripts/SLURM/02_pilot_run.sh

# v4 pilot — A100 interactive (single-batch overfit sanity check)
.PHONY: pilot-v4
pilot-v4:
	$(SRUN_A100) bash -c 'cd $(CURDIR) && uv run python scripts/pilot_run.py'

# v4 full pilot — all 3 stages, sbatch on A100
.PHONY: pilot-v4-all
pilot-v4-all:
	sbatch scripts/SLURM/02_pilot_v4.sh

# v5 pilot — A100 interactive, all v5 flags ON. Targets the v5 cache at
# data/processed_v5/; expects that directory to be populated via
# `uv run python scripts/preprocess_v5.py` first. The aux head must drive
# train/distogram_top1 > 0.5 inside 200 steps for the wiring to be correct.
.PHONY: pilot-v5
pilot-v5:
	$(SRUN_A100) bash -c 'cd $(CURDIR) && uv run python scripts/pilot_run.py \
		++paths.processed_dir=data/processed_v5 \
		++data.ligand_context_atoms=50 \
		++data.ligand_rich_features=false \
		++data.ligand_bond_topology=false \
		++model.distogram_aux_weight=0.2 \
		++model.distogram_num_bins=38'


# ==============================================================================
# TRAINING
# ==============================================================================
.PHONY: train
train:
	sbatch scripts/SLURM/03_train_model.sh

# v5 full curriculum — chains stage 1 → stage 2 → stage 3 via afterok
# dependencies so a single `make train-v5` queues the whole 3-stage run.
# Requires data/processed_v5/ populated (see scripts/preprocess_v5.py).
.PHONY: train-v5
train-v5:
	@J1=$$(sbatch --parsable scripts/SLURM/04a_v5_train_stage1.sh) ; \
	  echo "Stage 1 jobid: $$J1" ; \
	  J2=$$(sbatch --parsable --dependency=afterok:$$J1 scripts/SLURM/04b_v5_train_stage2_ddp.sh) ; \
	  echo "Stage 2 jobid: $$J2 (afterok:$$J1)" ; \
	  J3=$$(sbatch --parsable --dependency=afterok:$$J2 scripts/SLURM/04c_v5_train_stage3_ddp.sh) ; \
	  echo "Stage 3 jobid: $$J3 (afterok:$$J2)"


# ==============================================================================
# INFERENCE
# ==============================================================================
# Default inference target — set PDB=path/to/file.pdb on the command line
PDB ?= inputs/target.pdb

.PHONY: inference
inference:
	$(SRUN_GPU) bash -c 'cd $(CURDIR) && uv run uma-inverse design \
		--pdb $(PDB) \
		--config $(CONFIG) \
		--ckpt $(CKPT)'

# Inference shorthand: make infer PDB=my_structure.pdb
.PHONY: infer
infer: inference


# ==============================================================================
# BENCHMARK — submit the full paper-metric pipeline as sbatch.
# Override via sbatch --export; see scripts/SLURM/05_benchmark.sh
# for the full set of knobs.
# ==============================================================================

.PHONY: benchmark
benchmark:
	sbatch scripts/SLURM/05_benchmark.sh


# ==============================================================================
# SERVING (CPU service — local / Docker / HF Spaces, NOT the SLURM cluster)
# ==============================================================================
# These run on a normal machine (laptop, container, or the Space), not on the
# HPC cluster, so they are plain commands — no srun wrapping.

PORT ?= 7860

# The canonical HF weights are now the v5 model (see src/inference/weights.py),
# so `make serve` auto-fetches them on first run — no checkpoint flag needed
# (config.yaml is already the v5 arch). Pin a local file for speed/offline use:
#   make serve UMA_CKPT=checkpoints/uma-inverse-v5.ckpt
UMA_CKPT ?=

.PHONY: serve
serve:
	UMA_CKPT=$(UMA_CKPT) uv run --extra serving uvicorn src.serving.app:app --host 0.0.0.0 --port $(PORT)

.PHONY: docker-build
docker-build:
	docker build -t uma-inverse-serving .

.PHONY: docker-run
docker-run:
	docker run --rm -p $(PORT):7860 uma-inverse-serving

.PHONY: mcp
mcp:
	uv run --extra serving python -m src.mcp.server

# Precompute bundled-example results (loads the model → run where torch lives).
.PHONY: precompute-examples
precompute-examples:
	$(SRUN_CPU) bash -c 'cd $(CURDIR) && UMA_MAX_RESIDUES=1000 UMA_CKPT=$(UMA_CKPT) uv run python scripts/precompute_examples.py'


# ==============================================================================
# TESTING & LINTING (CPU only — no GPU needed)
# ==============================================================================
.PHONY: test
test:
	$(SRUN_CPU) bash -c 'cd $(CURDIR) && uv run --extra dev --extra serving python -m pytest tests/ -v --tb=short'

.PHONY: test-fast
test-fast:
	$(SRUN_CPU) bash -c 'cd $(CURDIR) && uv run --extra dev --extra serving python -m pytest tests/ -v --tb=short -x'

.PHONY: lint
lint:
	$(SRUN_CPU) bash -c 'cd $(CURDIR) && uv run --extra dev --extra serving ruff check src/ scripts/ tests/'

.PHONY: lint-fix
lint-fix:
	$(SRUN_CPU) bash -c 'cd $(CURDIR) && uv run --extra dev --extra serving ruff check --fix src/ scripts/ tests/'


# ==============================================================================
# ENV UTILITIES
# ==============================================================================

# Verify GPU, CUDA, and torch setup
.PHONY: env-check
env-check:
	$(SRUN_GPU) bash -c 'cd $(CURDIR) && uv run python -c "\
import torch; \
print(\"torch:\", torch.__version__); \
print(\"CUDA available:\", torch.cuda.is_available()); \
print(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"); \
print(\"bf16 supported:\", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"'

# List jobs you currently have queued/running
.PHONY: jobs
jobs:
	squeue -u $$USER
