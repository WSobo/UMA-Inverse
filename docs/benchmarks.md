# UMA-Inverse benchmark suite

A single `uma-inverse benchmark` invocation produces every table and
figure referenced in the paper — no post-processing required for the
standard analyses. This document is the reference for what that command
produces and how to run it.

## Quickstart

```bash
# Dev iteration (500 random PDBs from the val split; ~30 min on A5500)
uv run uma-inverse benchmark \
    --ckpt checkpoints/uma-inverse-best.ckpt \
    --val-json LigandMPNN/training/valid.json \
    --pdb-dir data/raw/pdb_archive \
    --out-dir outputs/benchmark

# Full validation split (~7153 PDBs; several hours on A5500)
uv run uma-inverse benchmark \
    --ckpt checkpoints/uma-inverse-best.ckpt \
    --val-json LigandMPNN/training/valid.json \
    --pdb-dir data/raw/pdb_archive \
    --all

# SLURM (uses A5500, 24h wall; override via BENCH_* env vars)
sbatch scripts/SLURM/05_benchmark.sh
sbatch --export=ALL,BENCH_N=all scripts/SLURM/05_benchmark.sh
```

## What gets computed

The pipeline runs three passes:

1. **Teacher-forced validation evaluation** — one forward pass per PDB
   with the native sequence as the autoregressive context, left-to-right
   decoding order. Produces sequence recovery, perplexity, entropy, and
   the full per-position probability distribution. This is the standard
   "sequence recovery" protocol used by every inverse-folding paper.
2. **Ligand-context ablation** — same evaluation with the ligand atom
   features zeroed. Paired with pass 1 by `pdb_id` so every row in
   `ablation_ligand.csv` is a within-PDB comparison.
3. **Temperature / diversity sweep** — autoregressive sampling at each
   requested T, measuring recovery against native + mean pairwise Hamming
   diversity within sample groups.

## Output layout

```
outputs/benchmark/<run_name>/
├── run_manifest.json          # reproducibility: git hash, ckpt sha256, config
├── summary.md                 # headline numbers + per-AA table in Markdown
├── summary.json               # same, machine-readable
├── per_pdb.csv                # one row per val PDB (recovery, perplexity, size)
├── per_position.parquet       # one row per residue (~1M rows for full val)
├── confusion_matrix.csv       # 20×20 counts, rows = native, cols = predicted
├── calibration.csv            # 10-bin reliability table + ECE
├── aa_composition.csv         # per-AA: native freq, predicted freq, recovery
├── ablation_ligand.csv        # ligand-aware vs masked, paired per PDB
├── temperature_sweep.csv      # per-T: mean recovery, std, diversity, confidence
└── figures/
    ├── confusion.png
    ├── calibration.png
    ├── near_ligand.png
    ├── aa_composition.png
    ├── perplexity_by_length.png
    └── temperature_diversity.png
```

Every file is self-describing — the CSV headers, JSON keys, and figure
titles are all production-grade. Paste `summary.md` directly into a lab
notebook or paper draft.

## Flag reference

**Inputs**

| Flag | Description |
|---|---|
| `--ckpt PATH` | Trained `.ckpt`. |
| `--val-json PATH` | LigandMPNN-style JSON list of PDB IDs (e.g. `LigandMPNN/training/valid.json` or `test_small_molecule.json`). |
| `--pdb-dir PATH` | Directory tree containing parsed PDB files (`<pdb_dir>/<xx>/<id>.pdb` or flat `<pdb_dir>/<id>.pdb`). |
| `--config PATH` | Hydra config matching the checkpoint's architecture. |

**Scope**

| Flag | Default | Description |
|---|---|---|
| `--n-pdbs N` | 500 | Cap on randomly sampled val PDBs. Use `--all` for the full split. |
| `--all` | off | Evaluate every resolvable PDB in `--val-json`. Supersedes `--n-pdbs`. |
| `--run-name NAME` | `<ckpt-stem>-<timestamp>` | Subdirectory name under `--out-dir`. |
| `--seed N` | 0 | Controls subsample + sampling RNG for reproducibility. |
| `--max-total-nodes N` | config | Residue-crop cap. For benchmarking, pass something large (≥5000) so structures aren't silently truncated. |

**Sweeps**

| Flag | Default | Description |
|---|---|---|
| `--skip-ablation` | off | Drop the ligand ablation (halves runtime). |
| `--skip-temperature` | off | Drop the temperature sweep (saves the most time). |
| `--temperatures "T1,T2,..."` | `0.0,0.1,0.2,0.5,1.0` | Sampling temperatures for the diversity curve. |
| `--samples-per-pdb N` | 3 | Samples at each T per PDB (≥2 required for Hamming diversity). |

**Environment**

| Flag | Description |
|---|---|
| `--device {cuda,cpu,auto}` | Default `auto`. |
| `-v` / `-vv` | INFO / DEBUG logging. |

## How each metric is computed

### Sequence recovery

```
recovery = correct_predictions / valid_positions
```

`correct_predictions` = positions where argmax of the model's logits
matches the native token. `valid_positions` excludes X (unknown AA). The
teacher-forced evaluation uses the native sequence in the AR context — a
common simplification of true AR sampling that every inverse-folding
paper uses for benchmarking.

Two variants are reported:

- **Pooled** — every valid position across every PDB. Treats long PDBs
  as having more weight.
- **Per-PDB mean ± std** — one recovery per PDB, then aggregate. Treats
  all PDBs equally regardless of size. This is the number to cite when
  comparing to LigandMPNN's published 0.52 (they report per-PDB mean).

### Perplexity

`exp(-mean(log_p over valid positions))`. Lower = better. Uniform guesses
over 20 AAs give 20.0. Reported in natural-log units (nats).

### Per-AA recovery

Row-wise diagonals of the confusion matrix normalised by native
frequency. "When the native is W, the model recovers W 82% of the time."
Useful for identifying classes the model struggles with (usually rare
ones like W, M).

### Expected calibration error (ECE)

Bucket predictions by predicted probability (10 equal-width bins), then
compute weighted mean of `|predicted - observed accuracy|` across bins.
A perfectly calibrated model has ECE = 0. Published sequence models
typically land in the 0.03–0.08 range.

### Ligand ablation (Δ recovery)

Two passes through the validation set — one with ligand features
enabled, one with `mask_ligand=True`. Delta is per-PDB so paired
t-tests work. Positive Δ means ligand context improves prediction (the
headline claim of any ligand-conditioned model).

### Near-ligand recovery

The per-position parquet has `distance_to_ligand` (Å to nearest ligand
heavy atom). `figures/near_ligand.png` buckets positions into distance
bins and plots recovery per bucket. A ligand-conditioned model should
show *higher* recovery near the ligand than far from it — this is the
most intuitive single plot for a paper's main figure.

### Temperature / diversity sweep

At each T, sample `--samples-per-pdb` sequences per PDB. Compute:

- **Recovery**: argmax sequence vs native.
- **Mean Hamming diversity**: mean pairwise fraction-of-differing
  positions within each PDB's samples. 0.0 at T=0 (identical samples);
  rises with T.

Plot both on a dual-axis line chart. The operating point "best
recovery-diversity trade-off" is where the curves cross.

## Interpreting the output

The **headline** block in `summary.md` is the single source of truth for
paper-level numbers. Cite those; everything else in the run directory is
provenance and figures.

A quick sanity check after every run:

```bash
jq '.headline | {recovery: .overall_recovery, perplexity, ece: .expected_calibration_error, n: .num_pdbs}' \
    outputs/benchmark/<run>/summary.json
```

Expected ranges for a reasonably trained UMA-Inverse on the LigandMPNN
valid split (informed by related work):

- overall_recovery: 0.50–0.60
- perplexity (nats): 2.5–5.5
- ECE: 0.02–0.08
- Δ recovery (ligand vs masked): +0.02–+0.10

Numbers well outside these ranges usually signal:

- Much lower recovery → checkpoint is wrong / model undertrained / val
  split mismatch.
- ECE > 0.15 → model is mis-calibrated (often from softmax temperature
  collapse during training).
- Δ < 0 → ligand features are doing *nothing* or are being fed in
  broken; inspect `load_structure(mask_ligand=...)`.

## Reproducibility

Every run writes `run_manifest.json` with:

- Full CLI command (`sys.argv`).
- Git hash at the time of the run.
- Checkpoint SHA-256 (so you can tell when the ckpt changed vs. only the
  run config).
- Full resolved Hydra config snapshot.
- Hostname, Python version, torch version.
- Start / stop timestamps.

Replaying a run is `uv run uma-inverse benchmark` with the same flags
— the manifest's `command` field is the literal line to use.

## Not included (intentionally)

Defer these until the core numbers are reported:

- **LigandMPNN head-to-head** — requires setting up their inference
  independently and reconciling parser differences. Skip unless a
  reviewer asks.
- **DSSP / SASA breakdowns** — needs external deps; would live in a
  separate `scripts/eval_structure_features.py` if added later.
- **Round-trip folding** (fold predicted sequence with AlphaFold, RMSD
  to input) — would need ~TB of AF2 inference; out of scope for a single
  A5500 benchmark run.
