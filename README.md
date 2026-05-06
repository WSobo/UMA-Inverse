# UMA-Inverse

**Ligand-conditioned protein inverse folding via dense pair-wise attention.**

Given a fixed protein–ligand backbone, UMA-Inverse predicts per-residue amino-acid identity (20 AA + X) under optional design constraints. The architecture is a single dense PairMixer encoder over the union of residue and ligand-atom nodes — no KNN sparsification — providing every residue a direct edge to every ligand atom. Built and trained on the LigandMPNN data protocol (parser, train/valid/test splits) so numbers are directly comparable.

→ Preprint: [docs/v2_preprint.md](docs/v2_preprint.md) (architecture, benchmarks, pocket-fixed redesign experiment)

---

## Install

Requires Python 3.13 and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url> UMA-Inverse
cd UMA-Inverse
uv sync                        # installs the package + deps from uv.lock
```

Then download the trained weights from [Hugging Face](https://huggingface.co/WSobo/UMA-Inverse_v2):

```bash
uv run python scripts/download_weights.py
```

After download, the canonical checkpoint lives at:

```
checkpoints/uma-inverse-v2.ckpt
```

GPU is recommended (any CUDA-capable card with ≥8 GB; the v2 model fits comfortably on a single A5500). CPU inference works but is ~50× slower.

---

## Quickstart

Design 10 sequences for a PDB at `T = 0.1` with random decoding order (matches the LigandMPNN inference protocol):

```bash
uv run uma-inverse design \
    --pdb my_complex.pdb \
    --ckpt checkpoints/uma-inverse-v2.ckpt \
    --num-samples 10 \
    --temperature 0.1 \
    --seed 42
```

Outputs are written under `outputs/<pdb-stem>-<timestamp>/`:

```
outputs/my_complex-2026-05-06T14-30-00/
├── my_complex.fa          # FASTA with native + 10 designs
├── ranked.csv             # per-design recovery, log-likelihood, dedup'd & ranked
└── run.log                # parameters used + per-PDB timing
```

---

## Inference reference

UMA-Inverse exposes two subcommands: `design` (sample sequences) and `score` (compute per-residue log-likelihoods under the trained model).

### `uma-inverse design`

**Sampling controls:**

| Flag | Default | Notes |
|---|---|---|
| `--num-samples` | 1 | Number of sequences to sample per PDB. |
| `--temperature` | 0.1 | Sampling temperature. `0.0` = argmax. |
| `--top-p` | (off) | Nucleus threshold in `(0, 1]`. |
| `--seed` | random | Base seed; sample `i` uses `seed + i`. |
| `--decoding-order` | `random` | `random` (matches LigandMPNN) or `left-to-right`. |
| `--batch-size` | 1 | Samples decoded in parallel per forward pass. GPU-memory dial. |

**Constraint flags** (all optional, all stackable):

| Flag | Format | Effect |
|---|---|---|
| `--fix` | `"A1 A2 B42C"` or `"A1,A2"` | Hold these residues at native. |
| `--redesign` | same | Redesign only these (complement is held native). |
| `--design-chains` | `"A,B"` | Redesign only these chains. |
| `--parse-chains` | `"A,B"` | Only parse these chains into the structure. |
| `--bias` | `"W:3.0,A:-1.0"` | Global per-AA logit bias. |
| `--bias-file` | JSON path | Per-residue bias: `{"A23": {"W": 3.0}, ...}`. |
| `--omit` | `"CDFG"` or `"C,D,F,G"` | Globally forbid these AAs. |
| `--omit-file` | JSON path | Per-residue omit: `{"A23": "CDFG", ...}`. |
| `--tie` | `"A1,A10\|B5,B15"` | Tie residue groups (groups separated by `\|`). |
| `--tie-weights` | matched to `--tie` | Per-position weights within tie groups. |

**Pocket-fixed redesign** (the use case characterized in the preprint):

```bash
uv run uma-inverse design \
    --pdb my_complex.pdb --ckpt checkpoints/uma-inverse-v2.ckpt \
    --fix "A12 A14 A56 A89 A124" \
    --num-samples 20 --temperature 0.1
```

**Batch mode** (many PDBs, with crash recovery):

```bash
# spec.json: {"path/to/a.pdb": {}, "path/to/b.pdb": {"fix": "A1 A2"}, ...}
uv run uma-inverse design \
    --pdb-list spec.json \
    --ckpt checkpoints/uma-inverse-v2.ckpt \
    --out-dir outputs/screen \
    --num-samples 5 \
    --resume    # skip PDBs already recorded in .done.txt
```

### `uma-inverse score`

Compute autoregressive log-likelihoods of the native (or a user-supplied) sequence at every position:

```bash
# Score the native sequence (averaged over 10 random decoding orders)
uv run uma-inverse score \
    --pdb my_complex.pdb \
    --ckpt checkpoints/uma-inverse-v2.ckpt \
    --mode autoregressive \
    --num-batches 10

# Score a custom sequence
uv run uma-inverse score \
    --pdb my_complex.pdb \
    --ckpt checkpoints/uma-inverse-v2.ckpt \
    --sequence "MKVL...QED"

# Single-AA scoring (each position scored with all others held native)
uv run uma-inverse score --pdb my_complex.pdb --ckpt ... --mode single-aa
```

Writes `scores_<pdb>.csv` (per-position log-likelihoods) and `scores_<pdb>.json` (summary).

### Other useful flags

- `--mask-ligand` — zero ligand-atom features (ablation, no ligand context).
- `--ligand-cutoff` — Å cutoff for ligand-proximal scoring (default 8.0).
- `--max-total-nodes` — cap residues+ligand atoms (overrides config; useful for OOM-tight GPUs).
- `--save-probs` — also dump the full `[N, L, 21]` probability tensor as `.npz`.
- `--device {cuda,cpu,auto}` — defaults to auto.
- `-v` / `-vv` — INFO / DEBUG logging.

For the full option list: `uv run uma-inverse design --help` or `--score --help`.

---

## Architecture & benchmarks

UMA-Inverse uses a stack of PairMixer blocks (triangle multiplication outgoing/incoming + transition MLP) over a dense `[L+M, L+M]` pair tensor built from RBF-encoded inter-atom distances. Decoding is autoregressive over a randomized residue order at `T = 0.1`, matching the LigandMPNN inference protocol.

Standard interface-recovery benchmark (LigandMPNN test splits, 10 designs per PDB, sidechain ≤ 5 Å cutoff):

| Split | UMA-v2 | LigandMPNN | ProteinMPNN |
|---|---:|---:|---:|
| Metal | 0.486 | 0.775 | 0.406 |
| Small molecule | 0.538 | 0.633 | 0.505 |

The preprint characterizes a regime — **pocket-fixed redesign** — where UMA-Inverse's dense attention shows a structural advantage that local KNN message-passing does not capture; see [docs/v2_preprint.md](docs/v2_preprint.md) §3.4–3.5 for details.

---

## Development

For training, data prep, and reproducing benchmark numbers:

- [docs/architecture.md](docs/architecture.md) — model internals
- [docs/inference.md](docs/inference.md) — extended CLI reference
- [docs/benchmarks.md](docs/benchmarks.md) — benchmark protocol
- [scripts/paper/](scripts/paper/) — reproduce the bioRxiv preprint experiments

```bash
# Reproduce training (HPC cluster, 8× A5500, ~3.8 days for stage 3)
sbatch scripts/SLURM/01a_fetch_data.sh        # fetch PDBs from RCSB
sbatch scripts/SLURM/01b_preprocess.sh        # cache .pt tensors
sbatch scripts/SLURM/02_pilot_run.sh          # 1-batch overfit sanity
sbatch scripts/SLURM/03_train_model.sh        # 3-stage curriculum
```

```bash
make test          # CPU-only pytest suite
make lint          # ruff
```

---

## Citation

If you use UMA-Inverse in your work, please cite the preprint:

```
Sobolewski, W. (2026). UMA-Inverse: Dense Pair-Wise Attention for
Ligand-Conditioned Protein Sequence Design. bioRxiv (TBD).
```

License: see [LICENSE](LICENSE).
