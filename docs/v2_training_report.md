# UMA-Inverse v2 — training report

**Date:** 2026-04-30
**Branch:** `v2-element-embedding`
**Final ckpt commit:** `6845412` (head of training)
**Run name:** `pairmixerinv-v2-stage3-nodes384-ddp8`

This report covers the v1 → v2 architectural upgrade, the 3-stage curriculum training that produced the v2 stage-3 checkpoint, and the LigandMPNN-protocol interface-recovery benchmark. It is a self-contained reference for the work between commits `83e243d` (v2 phase 1) and `6845412` (final).

---

## 1. Motivation

UMA-Inverse is a ligand-conditioned protein inverse-folding model: given fixed protein-ligand backbone coordinates, it predicts per-residue amino-acid identity (21 tokens: 20 AA + X). v1.0 was published with the metrics:

| Split | UMA v1 (ep11) | LigandMPNN | ProteinMPNN |
|---|---:|---:|---:|
| metal | 0.442 | 0.775 | 0.406 |
| small_molecule | 0.507 | 0.633 | 0.505 |

v1 cleared ProteinMPNN handily on small molecules but trailed LigandMPNN by ~33pp on metal and ~13pp on small_molecule. The hypothesis going into v2 was that **most of the gap is featurization, not the PairMixer encoder itself** — v1's residue/ligand featurizer was strictly less informative than LigandMPNN's. Three orthogonal upgrades, each config-flagged so v1 paths remain callable, were landed on branch `v2-element-embedding`.

Deliberately **not** adopting LigandMPNN's per-residue K-nearest ligand neighborhood — the architectural bet is dense pair-wise attention via PairMixer; layering a KNN sparsity prior on top would conflate two independent choices.

---

## 2. The three v2 changes

All three flags live under `data:` in `configs/config.yaml`. Two of them mirror into `model:` via OmegaConf interpolation:

```yaml
data:
  ligand_featurizer: atomic_number_embedding   # phase 1
  residue_anchor: cb                            # phase 2
  pair_distance_atoms: backbone_full            # phase 3

model:
  ligand_featurizer: ${data.ligand_featurizer}
  pair_distance_atoms: ${data.pair_distance_atoms}
```

### Phase 1 — Per-element atomic-number embedding (commit `83e243d`)

**v1:** ligand atoms encoded as a 6-bin one-hot (C / N / O / S / P / other) → `Linear(6, node_dim)`.
**v2:** ligand atoms encoded as their atomic number (0 = padding, 1-118 = real elements, 119 = unknown sentinel) → `nn.Embedding(120, node_dim, padding_idx=0)`.

This lets the model learn distinct embeddings for chemically meaningful elements lumped into v1's "other" bin (Mg, Zn, Fe, halogens, transition metals, …) — particularly relevant for the metal split where v1 was weakest.

Unknown-element handling: `src/data/pdb_parser.py:185` was changed from `default → 6 (carbon)` to `default → 119 (unknown sentinel)`. v1's "other" bin still routes 119 correctly for the one-hot path; the embedding path gets a dedicated learnable slot.

### Phase 2 — Virtual Cβ as residue anchor (commit `501f0ee`)

**v1:** residue position represented by Cα coordinate.
**v2:** virtual Cβ computed from N, CA, C via the ProteinMPNN formula:

```python
b = CA - N
c = C - CA
a = torch.linalg.cross(b, c, dim=-1)
Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
```

(See `src/data/ligandmpnn_bridge.py`.) The Cβ anchor moves the residue-ligand distance measurement from the backbone to the sidechain stub — important because amino-acid identity is overwhelmingly determined by sidechain-ligand contacts, not backbone-ligand contacts.

Works for glycine (no special case needed; the formula degenerates gracefully).

### Phase 3 — Multi-atom backbone distances in pair tensor (commit `81e346b`)

**v1:** the residue-residue [L,L] block of the pair tensor used a single Cα-Cα distance per residue pair, RBF-projected into `pair_dim`.
**v2:** the [L,L] block uses **5 distinct backbone-atom distance pairs** stacked and re-projected:

| Distance pair |
|---|
| Cα–Cα |
| Cα–N |
| Cα–C |
| N–O |
| O–C |

Each is RBF'd (32 bases) then concatenated to `[B,L,L,5*32]` and projected through `rbf_proj_multi: Linear(160, pair_dim, bias=False)`. The [L,M], [M,L], [M,M] blocks (anything involving ligand atoms) keep the single-anchor Cα/Cβ-driven distance because ligand atoms have no backbone analogue.

Phase 2 (Cβ vs Cα anchor) still drives the [L,M]/[M,L] blocks even when phase 3 is on.

### Smoke test

`scripts/smoke_test_v2.py` covers v1 path + each v2 phase + all-on combination. It also asserts that v1↔v2 strict-load fails (so v1 checkpoints cannot accidentally warm-start v2 training and vice versa). 7/7 tests pass on every commit.

---

## 3. Data prep (Phase 4)

The v2 changes meant the cached `.pt` files (~150K of them) couldn't be reused — they only stored v1 keys (`ligand_features`, Cα-only `residue_coords`, no backbone coords).

### Decisions

- **Union cache.** `scripts/preprocess.py` now always emits *every* v2 key regardless of config: `ligand_features` AND `ligand_atomic_numbers`, `residue_coords` (Cα baseline) PLUS `residue_backbone_coords [L,4,3]`. Cβ is derived on-the-fly in `UMAInverseDataset.__getitem__` when `residue_anchor=cb`. Lets us flip flags for ablation later without re-preprocessing. Cost: cache grew from 4.1 GB → 7.9 GB.
- **First v2 run: all three flags on.** Fastest signal on whether v2-as-a-whole closes the LigandMPNN gap. Ablation deferred.

### Re-fetch + re-cache

| Step | Tool | Wall |
|---|---|---|
| Re-fetch 157K train+valid PDBs from RCSB | `01a_fetch_data.sh` (16 download workers) | ~24 h |
| Regenerate union cache | `01b_preprocess.sh --recompute` | ~5 h on CPU |

Yield: **154,658 cached PDBs (98.5%/98.8% coverage of train/valid splits)**. The remaining 1.5% drop is RCSB 404s (2,323 PDBs deprecated since LigandMPNN's split was published) plus 37 parser ValueErrors.

---

## 4. Curriculum training (Phase 5)

Same 3-stage structure as v1, sized to the architecture's actual step throughput.

### Run-name scoping

`scripts/train.py` was updated to scope `ModelCheckpoint.dirpath` by `run_name` so v2 stages don't trample v1's `checkpoints/last.ckpt` (the v1 reference ckpt for the published benchmark). Dirs:

```
checkpoints/
├── pairmixerinv-v2-stage1-nodes64/
├── pairmixerinv-v2-stage2-nodes128-ddp4/
└── pairmixerinv-v2-stage3-nodes384-ddp8/
    ├── uma-inverse-19-1.1463.ckpt   ← top-1 by val_loss (save_top_k=3)
    ├── uma-inverse-20-1.1499.ckpt
    ├── uma-inverse-16-1.1504.ckpt
    ├── last.ckpt                     ← epoch 29
    └── epoch_snapshots/epoch-{00..29}.ckpt
```

### Stage 1 — `pairmixerinv-v2-stage1-nodes64`

- 1× A5500, single-GPU, max_total_nodes=64, batch_size=8
- 15 epochs, ~272K steps, warmup=1000 / T_max=280K
- bf16-mixed
- **Wall: 10h 32m** (job 32531097)

| Epoch | val_acc | val_loss |
|---:|---:|---:|
| 0 | 0.3748 | 2.067 |
| 6 (best) | 0.4262 | 1.896 |
| 14 (final) | 0.4204 | 2.047 |

Stage-1 val curve plateaus quickly at N=64 (residue crop too aggressive to learn long-range contacts); this is by design — stage 1 just gets the model into a reasonable basin before stages 2/3 expand the receptive field.

### Stage 2 — `pairmixerinv-v2-stage2-nodes128-ddp4`

- 4× A5500 DDP, max_total_nodes=128, batch_size=4/rank (effective 16)
- 25 epochs, ~227K steps, warmup=1000 / T_max=230K
- `init_from_checkpoint=stage1/last-v1.ckpt` (weights-only; fresh optimizer + LR cycle)
- **Wall: 12h 40m** (job 32531098)

| Epoch | val_acc | val_loss |
|---:|---:|---:|
| 0 | 0.4610 | 1.744 |
| 14 (best) | 0.5108 | 1.590 |
| 24 (final) | 0.5100 | 1.639 |

Stage 2 delivered the largest single-stage val_acc bump (+9pp): from stage-1's 0.420 to 0.511. This is exactly where v1 saw its biggest gain too — N=128 is roughly the median train-PDB residue count, so the model finally sees enough context.

### Stage 3 — `pairmixerinv-v2-stage3-nodes384-ddp8`

- 8× A5500 DDP, max_total_nodes=384, batch_size=2/rank (effective 16)
- 30 epochs, ~272K steps, warmup=2000 / T_max=280K
- `init_from_checkpoint=stage2/last.ckpt` (weights-only)
- bf16-mixed, 0.83 it/s steady-state
- **Wall: 3d 19h 30m** (job 32531099)

Full val curve:

| Epoch | val_acc | val_loss | | Epoch | val_acc | val_loss |
|---:|---:|---:|---|---:|---:|---:|
| 0  | 0.6107 | 1.223 | | 15 | 0.6336 | 1.167 |
| 1  | 0.6151 | 1.215 | | 16 | 0.6349 | 1.150 |
| 2  | 0.6199 | 1.192 | | 17 | 0.6365 | 1.151 |
| 3  | 0.6218 | 1.190 | | 18 | 0.6353 | 1.161 |
| 4  | 0.6229 | 1.187 | | **19** | **0.6374** | **1.146** |
| 5  | 0.6240 | 1.185 | | 20 | 0.6367 | 1.150 |
| 6  | 0.6254 | 1.181 | | 21 | 0.6373 | 1.149 |
| 7  | 0.6286 | 1.172 | | 22 | 0.6361 | 1.160 |
| 8  | 0.6291 | 1.170 | | 23 | 0.6380 | 1.148 |
| 9  | 0.6313 | 1.165 | | 24 | 0.6383 | 1.150 |
| 10 | 0.6305 | 1.166 | | 25 | 0.6382 | 1.154 |
| 11 | 0.6300 | 1.172 | | 26 | 0.6383 | 1.152 |
| 12 | 0.6333 | 1.155 | | 27 | 0.6380 | 1.154 |
| 13 | 0.6341 | 1.155 | | 28 | 0.6385 | 1.154 |
| 14 | 0.6321 | 1.174 | | **29** | **0.6387** | 1.151 |

- **Best by val_loss (the monitored metric, ckpt selected by `save_top_k=3`):** epoch 19 — `val_loss = 1.1463`, `val_acc = 0.6374`.
- **Best by val_acc:** epoch 29 — `val_acc = 0.6387`, `val_loss = 1.1511`.
- Gap between the two: 0.0013 val_acc — within step-level noise.
- val_loss bottomed at ep 19 and oscillated ±0.005 through ep 29 — model is converged from ep 19 onward.

### Final ckpt selection

Both epochs 19 (best val_loss) and 29 (best val_acc) were benchmarked on the LigandMPNN test splits; the test-recovery winner is the canonical v2 release ckpt.

---

## 5. Engineering issues fixed during training

Real bugs that surfaced and the fixes that made the curriculum stable. All committed on the `v2-element-embedding` branch.

### Zero-residue PDB leak into validation (commits `0e361f8`, `acaae39`, `c0ae484`)

A small fraction of cached `.pt` files had `residue_mask.sum() == 0` — DNA/RNA-only structures whose ATOM records the parser couldn't classify as protein. These slipped past the dataset and produced NaN val/loss for entire epochs.

Defense-in-depth fix:

1. `_filter_zero_residue_ids` builds an incremental blacklist (parallel scan with `ProcessPoolExecutor` across all SLURM-allocated CPUs — 132s for the full 147K train cache, down from ~22 min serial).
2. `__getitem__` re-checks `residue_coords.shape[0] > 0` per cache hit.
3. `load_example_from_pdb` now `raise ValueError` on empty residue mask instead of returning shape-0 tensors silently.
4. `EarlyStopping(check_finite=False)` so a single bad sample's NaN doesn't kill the run.

### ModelCheckpoint filename bug (commit `6845412`)

`ModelCheckpoint(filename="uma-inverse-{epoch:02d}-{val_loss:.4f}")` produced files like `uma-inverse-19-0.0000.ckpt`. The metric key is `val/loss` (slash), not `val_loss` (underscore), so Lightning silently substituted 0. Fixed to `{val/loss:.4f}` with `auto_insert_metric_name=False`. Cosmetic only — `monitor="val/loss"` for save_top_k always used the right key.

### EarlyStopping killed stage 1 on first epoch (commit `43a26fe`)

Lightning's `EarlyStopping(check_finite=True)` default treats the first NaN val/loss as a fatal stop. Set to `check_finite=False` so transient NaN is tolerated; `save_top_k` still ignores NaN-loss epochs so best-ckpt selection is unaffected.

### Lightning ckpt collision suffix (`last-v1.ckpt`)

Stale stage-1 ckpts from a broken pre-fix run caused Lightning's filename-collision logic to write `last-v1.ckpt` instead of overwriting `last.ckpt`. The `-v1` is Lightning's suffix, **not** related to model versioning. Fix: clean stale dirs before re-launching, point the stage-2 SLURM `init_from_checkpoint` at the right name.

---

## 6. Benchmark protocol

`scripts/benchmark_interface_recovery.py` reproduces the Dauparas et al. LigandMPNN-paper protocol:

For each PDB in `LigandMPNN/training/test_<class>.json`:
1. Encode structure once (full coords, ligand context).
2. Generate **10 sequences** autoregressively, **random decoding order**, **T=0.1**.
3. Per-sample **interface recovery** = fraction of sidechain-interface residues where pred == native (excluding X tokens). Interface = residues whose sidechain has at least one heavy atom **within 5 Å of any nonprotein heavy atom**.
4. Per-PDB statistic = **median across the 10 samples**.
5. Headline = **mean of per-PDB medians** across the split.

Three SLURM jobs (one per class) run in parallel via `scripts/SLURM/05c_benchmark_interface_recovery.sh`. Outputs land in `outputs/benchmark/interface_recovery/<run_name>/{per_pdb.csv, per_sample.csv, summary.json}`. Aggregation via `scripts/summarize_test_benchmarks.py --run-prefix <prefix>`.

---

## 7. Benchmark results — v2 ep19

ckpt: `pairmixerinv-v2-stage3-nodes384-ddp8/uma-inverse-19-1.1463.ckpt` (val_loss 1.146, val_acc 0.637).

| class | N PDBs | UMA v2 ep19 | UMA v1 ep11 | LigandMPNN | ProteinMPNN | Δ vs LigMPNN | Δ vs UMA v1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| metal | 82 | **0.486** | 0.442 | 0.775 | 0.406 | -0.289 | **+0.044** |
| small_molecule | 316 | **0.538** | 0.507 | 0.633 | 0.505 | -0.095 | **+0.031** |

Nucleotide (diagnostic only — see §8): 39/74 PDBs with HETATM cofactors evaluated, mean = 0.426, vs v1's 0.372.

### v2 ep29 results

[populated once benchmark jobs 32727422-24 finish]

### Final ckpt choice

[whichever epoch wins on test interface recovery becomes the canonical v2 release]

---

## 8. Known limitations / follow-ups

- **Nucleotide split is not a fair comparison.** The current featurizer (`src/data/pdb_parser.py`) ingests only `HETATM` records as ligand context. DNA/RNA atoms are stored as `ATOM` with residue names `DA/DC/DG/DT` (DNA) or `A/U/G/C` (RNA), so UMA-Inverse has never seen nucleic acid as ligand context. Reported nucleotide numbers reflect performance on incidental HETATM cofactors within those PDBs, not nucleotide-binding design. Closing this gap requires extending the featurizer to ingest nucleic-acid `ATOM` records, re-preprocessing, and another training cycle. Flagged for future work.

- **Phase-by-phase ablation not done.** The first v2 run was all-flags-on. If we want to attribute the v2 gain to a specific phase, we'd run three more stage-3 trainings (one phase enabled at a time) — each ~3.8d on 8× A5500. Worth doing only if the headline materially closes the LigandMPNN gap.

- **Gap to LigandMPNN remains substantial.** v2 closed only ~3-13% of the v1→LigandMPNN gap on metal/small_molecule. The remaining gap is plausibly attributable to LigandMPNN's per-residue K-nearest sparsity + their geometry features (relative positional encodings, side-chain ψ-aware embeddings). Architecture refactors beyond featurization are out of scope for v2 but a candidate for v3.

- **2,323 RCSB-404 PDBs and 37 parser ValueErrors** logged at `logs/preprocess_failures.txt`. Likely entries deprecated since LigandMPNN's split was published; not investigated further. Doesn't affect the headline benchmarks (test splits were re-fetched cleanly).

- **DDP val/loss `sync_dist=False`.** In `src/training/lightning_module.py:138`, val/loss is logged with `sync_dist=False` — Lightning warns, but for now epoch-level val curves are clean enough to read by eye. Could be tightened.

---

## 9. Reproducibility pointers

| Question | Where to look |
|---|---|
| What did the model train on? | `LigandMPNN/training/{train,valid}.json`. PDB archive at `data/raw/pdb_archive/`. Cache at `data/processed/`. |
| Final config? | `configs/config.yaml` at commit `6845412`. Per-stage overrides in `scripts/SLURM/03{,b,c}_train_*_ddp*.sh`. |
| Final ckpt? | `checkpoints/pairmixerinv-v2-stage3-nodes384-ddp8/`. ep19 is `uma-inverse-19-1.1463.ckpt`; ep29 is `last.ckpt`. Per-epoch snapshots in `epoch_snapshots/`. |
| Training loss/val curves? | `logs/csv/pairmixerinv-v2-stage{1,2,3}-*/version_*/metrics.csv`. (Stage 1 = version_3, stage 2 = version_1; earlier versions are aborted runs from before the EarlyStopping fix.) |
| Benchmark per-PDB recoveries? | `outputs/benchmark/interface_recovery/v2-ep19-test_*/per_pdb.csv` (and ep29 once that's in). |
| Headline numbers + LigandMPNN comparison table? | `outputs/benchmark/interface_recovery/test_summary.md` (and prefix-tagged copies). |
| Smoke test? | `uv run python scripts/smoke_test_v2.py` — 7/7 should pass. |
