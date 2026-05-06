# UMA-Inverse Inference CLI

A production-grade command-line interface for running the trained UMA-Inverse
model. Ships with two subcommands:

- **`uma-inverse design`** — sample new sequences for one or many PDBs.
- **`uma-inverse score`** — compute per-position log-likelihoods under the
  trained model.

The CLI is a thin wrapper over the library in `src/inference/`. Every
feature surfaced here is also callable as Python — import from
`src.inference` if you'd rather drive the pipeline from a notebook.

---

## Installation and prerequisites

This is part of the UMA-Inverse repo. From the project root:

```bash
uv sync --extra dev           # installs all deps including typer, pytest
uv run uma-inverse --help     # top-level help
uv run uma-inverse design --help
uv run uma-inverse score --help
```

The CLI is registered in `pyproject.toml` under `[project.scripts]`, so
once `uv sync` runs, `uv run uma-inverse …` picks it up automatically.

On the HPC cluster **always** wrap GPU inference in `srun` or submit via
`sbatch` — never run bare `python` on the head node.

---

## `uma-inverse design`

### Quickstart: single PDB

```bash
uv run uma-inverse design \
    --pdb inputs/my_protein.pdb \
    --ckpt checkpoints/uma-inverse-v2.ckpt \
    --num-samples 10 \
    --temperature 0.1 \
    --top-p 0.95 \
    --seed 42 \
    --out-dir outputs/my_run
```

Writes:

```
outputs/my_run/<pdb-stem>-<timestamp>/
├── fastas/my_protein.fa              # 10 sampled sequences + native
├── confidences/my_protein.json       # per-residue confidence + top-3
├── ranked.csv                        # deduped samples sorted by mean NLL
└── run_manifest.json                 # git hash, config, seed, timings
```

### Quickstart: batch mode

Write a batch spec as JSON — one entry per PDB, optional per-PDB overrides:

```json
{
    "inputs/1BC8.pdb": {},
    "inputs/4GYT.pdb": {
        "fix": "A12 A13 A14",
        "bias": "W:3.0,A:-1.0"
    },
    "inputs/complex_with_drug.pdb": {
        "tie": "C10,D10|C20,D20",
        "mask_ligand": false
    }
}
```

Run once, interrupt, resume:

```bash
uv run uma-inverse design --pdb-list my_spec.json --ckpt ckpt.ckpt \
    --out-dir outputs/screen --num-samples 5 --seed 0

# If killed by SLURM time limit, pick up where you left off:
uv run uma-inverse design --pdb-list my_spec.json --ckpt ckpt.ckpt \
    --out-dir outputs/screen --num-samples 5 --seed 0 \
    --run-name <same-name-as-before> --resume
```

`<run-dir>/.done.txt` tracks completed PDBs line-by-line; `--resume` skips
any PDB whose path appears there.

### Flag reference

**Inputs**

| Flag | Description |
| --- | --- |
| `--pdb PATH` | Single PDB file. Mutually exclusive with `--pdb-list`. |
| `--pdb-list PATH` | JSON batch spec `{pdb_path: {overrides}}`. |
| `--ckpt PATH` | Trained `.ckpt` (Lightning state dict auto-stripped). |
| `--config PATH` | Hydra config with model hyperparameters (default: `configs/config.yaml`). |

**Sampling**

| Flag | Default | Description |
| --- | --- | --- |
| `--num-samples N` | 1 | Sequences per PDB. |
| `--batch-size N` | 1 | Parallel forward passes (GPU memory dial). |
| `--temperature T` | 0.1 | `0.0` = argmax; higher = more diversity. |
| `--top-p P` | unset | Nucleus threshold in `(0, 1]`. Disables when omitted. |
| `--seed N` | random | Deterministic reproduction (sample *i* uses `seed + i`). |
| `--decoding-order {random,left-to-right}` | random | `left-to-right` only for debugging. |

**Residue selection**

All selectors accept chain-letter notation: `"A23"`, `"B42C"` (insertion
code), negative numbers for unusual PDBs, spaces or commas as separators.

| Flag | Description |
| --- | --- |
| `--fix "A1 A2 B42C"` | Residues held at native (never redesigned). |
| `--redesign "A5 A6 A7"` | Residues allowed to change; complement fixed. |
| `--design-chains "A,B"` | Chains whose residues count as designable. |
| `--parse-chains "A,B"` | Only parse these chains from the PDB. |

**Constraints**

| Flag | Description |
| --- | --- |
| `--bias "W:3.0,A:-1.0"` | Global AA bias added to logits. |
| `--bias-file FILE.json` | Per-residue bias: `{"A23": {"W": 3.0}}`. |
| `--omit "CDFG"` | Global AA ban (X is always banned). |
| `--omit-file FILE.json` | Per-residue ban: `{"A23": "CDFG"}`. |
| `--tie "A1,A10\|B5,B15"` | Tied groups (groups sep by `\|`). |
| `--tie-weights "0.5,0.5\|0.5,0.5"` | Weights aligned to `--tie`. |

**Structure parsing / ablations**

| Flag | Description |
| --- | --- |
| `--mask-ligand` | Zero ligand features — ablation for ligand-conditioned design. |
| `--ligand-cutoff F` | Å cutoff for ligand proximity (default: config, 8.0). |
| `--include-zero-occupancy` | Parse atoms with occupancy=0. |
| `--max-total-nodes N` | Cap on residues + ligand atoms after cropping. |

**Output**

| Flag | Default | Description |
| --- | --- | --- |
| `--out-dir DIR` | `outputs/` | Parent of the run directory. |
| `--run-name NAME` | stem+timestamp | Subdirectory name. |
| `--suffix TEXT` | empty | Appended to output file basenames. |
| `--save-probs` | off | Also dump `probs/<pdb>.npz` with full `[N, L, 21]` distribution. |
| `--ranked / --no-ranked` | ranked | Write dedup'd `ranked.csv`. |
| `--include-native / --no-native` | include | First FASTA record = native sequence. |

**Batch control**

| Flag | Description |
| --- | --- |
| `--resume` | Skip PDBs already listed in `<run-dir>/.done.txt`. |

**Environment**

| Flag | Description |
| --- | --- |
| `--device {cuda,cpu,auto}` | Default: `auto` (CUDA if available). |
| `-v` / `-vv` | Increase logging verbosity (INFO / DEBUG). |

### Output format details

#### `fastas/<pdb>.fa`

One record per sample, header follows LigandMPNN convention:

```
>1BC8 kind=native length=93
MDSAITLWQFLLQLLQKPQNKHMICWTSNDGQFKLLQAEEVARLWGIRKNKPNMNYDKLSRALRYYYVKNIIKKVNGQKFVYKFVSYPEILNM
>1BC8 sample=0 seed=42 T=0.1 overall_confidence=0.9371 top_p=0.95 ligand_confidence=0.9371
MDSKISLIEWL… (designed sequence)
```

- `overall_confidence` = `exp(mean(log_p) over redesigned residues)`.
- `ligand_confidence` = same restricted to ligand-neighbour residues.

#### `confidences/<pdb>.json`

Per-sample, per-residue distribution summary:

```json
{
  "pdb_id": "1BC8",
  "samples": [{
    "seed": 42,
    "overall_confidence": 0.9371,
    "positions": [
      {
        "position": 0, "residue_id": "C1",
        "sampled": "M", "sampled_prob": 1.0,
        "top_k": [{"aa":"M","prob":1.0}, {"aa":"R","prob":0.0}, …],
        "entropy": 0.0, "margin": 1.0
      },
      …
    ]
  }]
}
```

Useful downstream filters:
- `entropy` low + `margin` high → the model is confident about one AA.
- `entropy` high → position worth experimentally exploring.

#### `ranked.csv`

Dedup'd across samples, sorted by ascending mean NLL:

```
pdb_id,sequence,mean_nll,overall_confidence,ligand_confidence,sample_seeds
1BC8,MDSPISLIEWLAWWLSDP…,2.5721,0.9589,0.9589,1;7
1BC8,MDSKISLIEWLKWWLSDP…,2.8103,0.9371,0.9371,42
```

Seeds column lists every sample seed that produced that exact sequence.

#### `run_manifest.json`

Reproducibility record:

```json
{
  "run_name": "1bc8-20260422-123456",
  "command": "uma-inverse design --pdb … --ckpt …",
  "checkpoint_path": "checkpoints/uma-inverse-v2.ckpt",
  "checkpoint_sha256": "abc123…",
  "config_path": "configs/config.yaml",
  "config_snapshot": { /* full resolved config */ },
  "git_hash": "ee8c5c3a…",
  "model_revision": "…",   // git hash of training commit if discoverable
  "hostname": "phoenix-05",
  "python_version": "3.13.1",
  "torch_version": "2.5.1+cu124",
  "seed": 42,
  "start_timestamp": "2026-04-22T14:25:07",
  "stop_timestamp":  "2026-04-22T14:25:09",
  "temperature": 0.1, "top_p": 0.95,
  "num_pdbs": 1, "num_samples_per_pdb": 10
}
```

#### `probs/<pdb>.npz` (optional, `--save-probs`)

```python
>>> import numpy as np
>>> d = np.load("outputs/run/probs/1BC8.npz", allow_pickle=True)
>>> d["probs"].shape         # [num_samples, L, 21]
>>> d["token_ids"].shape     # [num_samples, L]
>>> d["log_probs"].shape     # [num_samples, L]
>>> d["seeds"].tolist()      # seeds used per sample
>>> d["residue_ids"].tolist()
```

---

## `uma-inverse score`

Compute per-position log-likelihoods.

```bash
uv run uma-inverse score \
    --pdb inputs/my_protein.pdb \
    --ckpt checkpoints/uma-inverse-v2.ckpt \
    --mode autoregressive \
    --num-batches 10 \
    --seed 42 \
    --out-dir outputs/score_native
```

### Modes

| `--mode` | What it computes |
| --- | --- |
| `autoregressive` | `p(AA_t | structure, AA_{<t})` averaged over `--num-batches` random decoding orders. |
| `single-aa` | `p(AA_t | structure, AA_{all except t})` — one forward pass per residue. |

### Flags

| Flag | Description |
| --- | --- |
| `--sequence AACCDD…` | Score this sequence instead of the native. Length must equal parsed residue count. |
| `--use-sequence / --no-use-sequence` | Feed the scored sequence into the AR context (default) or mask as all-X. |
| `--num-batches N` | AR mode only. More = less variance, more compute. |
| `--seed N` | Reproduces the random decoding orders. |

All the structure-parsing flags from `design` (`--parse-chains`,
`--mask-ligand`, `--ligand-cutoff`, `--include-zero-occupancy`,
`--max-total-nodes`) are also accepted here.

### Outputs

```
outputs/score_native/<stem>-score-<timestamp>/
├── scores_<pdb>.csv       # position, residue_id, aa, log_prob, prob
├── scores_<pdb>.json      # summary (mean_log_prob, sum_log_prob, mode)
└── run_manifest.json
```

---

## Programmatic API

Every CLI behaviour is available as Python. Typical pattern:

```python
from src.inference import (
    InferenceSession, DesignConstraints,
)
from src.inference.decoding import autoregressive_design, score_sequence
from src.inference.output import (
    write_samples_fasta, write_per_residue_confidence, build_ranked_rows,
)

session = InferenceSession.from_checkpoint(
    config_path="configs/config.yaml",
    checkpoint="checkpoints/uma-inverse-v2.ckpt",
    device="cuda",
)
ctx = session.load_structure("inputs/my.pdb", mask_ligand=False)
constraints = DesignConstraints.from_cli(fix="A1 A2 A3", bias="W:3.0")
resolved = constraints.resolve(ctx)

samples = autoregressive_design(
    session, ctx, resolved,
    num_samples=20, batch_size=5,
    temperature=0.1, top_p=0.95, seed=42,
)

write_samples_fasta(
    "out.fa", pdb_id="my", ctx=ctx, samples=samples,
    designable_mask=resolved.designable_mask.cpu(),
)
```

---

## Differences from LigandMPNN

UMA-Inverse's inference layer aims at LigandMPNN feature parity plus a
handful of quality-of-life additions.

**Ported, renamed for clarity.** Flag changes are listed in the plan;
the semantics follow LigandMPNN where applicable. Notable:
`--mask-ligand` replaces the inverted `--ligand_mpnn_use_atom_context 0`;
`--bias-file` / `--omit-file` accept the same JSON schemas as the
LigandMPNN `*_multi` counterparts; `--tie` replaces the split
`--symmetry_residues` / `--symmetry_weights` pair.

**Not ported.**

- Side-chain packing (OpenFold dep, different model).
- ProteinMPNN / SolubleMPNN / membrane variants (different models).
- `--homo_oligomer` shortcut (expressible via `--tie`).

**New in UMA-Inverse.**

- **Per-residue confidence JSON** — position-level entropy, top-3 AAs,
  margin-to-runner-up. LigandMPNN surfaces only two scalars.
- **Nucleus (top-p) sampling** — alternative to pure temperature.
- **Sample deduplication + ranking** — `ranked.csv` merges identical
  sequences across seeds and sorts by NLL.
- **Resumable batch runs** — `.done.txt` allows `--resume` to skip
  completed PDBs. Essential for long SLURM jobs.
- **`run_manifest.json`** — full reproducibility record: git hash,
  checkpoint sha256, config snapshot, hostname, timings. No more
  "what flags produced this FASTA?" questions after the fact.
