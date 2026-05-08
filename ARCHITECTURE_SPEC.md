# UMA-Inverse Architecture Spec (v3)

This document is the implementation contract for UMA-Inverse: a LigandMPNN-comparable inverse folding model using a dense invariant PairMixer encoder, trained on consumer-class (single A5500, 24 GB) hardware via a 3-stage curriculum.

**v3 thesis.** v2 trained the dense PairMixer encoder against a stripped-down feature set vs LigandMPNN's; v2-vs-LigandMPNN deltas could therefore reflect *features* rather than *architecture*. v3 adopts LigandMPNN's full feature set verbatim and holds PairMixer constant. v3-vs-LigandMPNN now tests architecture (dense pair tensor + triangle multiplication vs sparse KNN MPNN). v3-vs-v2 tests features.

## 1) Scope and constraints

- Objective: predict amino-acid identity per residue for a fixed protein-ligand structure.
- Inputs: protein backbone coordinates + ligand atom coordinates/types + (training-only) per-residue sidechain heavy atoms.
- Outputs: per-residue logits over 21 tokens (20 AA + X).
- Data compatibility: LigandMPNN split JSON files (`train.json`, `valid.json`).
- Hardware target: single A5500 24 GB for stages 1/pilot/inference; 4×–8×A5500 DDP for stages 2–3.
- Non-goals:
  - no full SE(3) kernel stack (rotational invariance comes from pairwise distances + frame-relative angles, not equivariant operators)
  - no auxiliary sidechain-coordinate output head (sidechains are *inputs*, not predictions)
  - no diffusion head
  - no migration to A100/H100 (consumer-hardware constraint is durable)

## 2) Repository layout

```
configs/                         # Hydra config surface (config.yaml is root)
src/
  data/
    pdb_parser.py                # vendored BioPython parser (no LigandMPNN runtime import)
    ligandmpnn_bridge.py         # parser → tensors, frame angles, sidechain plumbing
    datamodule.py                # Dataset + collate + sidechain-context aug
  models/
    pairmixer_block.py           # TriMul-out / TriMul-in / Transition
    uma_inverse.py               # full model: pair init, encoder, AR, decoder
  training/
    lightning_module.py          # train/val step, optimizer, schedule
scripts/
  preprocess.py                  # PDB → .pt union cache
  train.py / pilot_run.py        # Hydra entry points
  inference.py                   # PDB → FASTA
  SLURM/                         # sbatch wrappers; v3 stages = 02_pilot_v3, 04a/b/c
tests/                           # pytest, CPU-only
LigandMPNN/                      # REFERENCE ONLY — only training/{train,valid}.json used
```

## 3) Data pipeline

### 3.1 Parsing

`src.data.pdb_parser.parse_pdb` is the runtime parser — BioPython-based, no `LigandMPNN/data_utils.py` import. Output schema preserves LigandMPNN semantics:

- `S` token vocabulary: alphabetical 20 AA + X=20. Common modified residues (MSE→M, SEP→S, HSD/HSE/HSP/HIE/HID/HIP→H, etc.) are normalized.
- `mask`: residue valid (Cα present).
- `chain_mask`: design intent (default all-True at training; per-chain at inference).
- `Y`, `Y_t`, `Y_m`: ligand heavy atoms, atomic numbers, "near protein" mask (within `cutoff_for_score=8 Å` of any Cα).
- `sidechain_coords` `[K, 3]`, `sidechain_atomic_numbers` `[K]`, `sidechain_residue_idx` `[K]` — per-residue sidechain heavy atoms (everything except N/CA/C/O, hydrogens excluded). Indices align with the parsed residue list.

### 3.2 Node features

Protein nodes:
- Anchor coordinate: virtual Cβ (default, `residue_anchor='cb'`) constructed via the ProteinMPNN analytic formula `Cβ = -0.58273431·a + 0.56802827·b - 0.54067466·c + Cα` from N/Cα/C; or Cα (`'ca'`, legacy).
- 6D dihedral feature: `[sin φ, sin ψ, sin ω, cos φ, cos ψ, cos ω]`.

Ligand nodes — three featurizers (mutually exclusive):
- `ligandmpnn_atomic` (v3 default): one-hot(atomic_num=0..119) + one-hot(group=0..18) + one-hot(period=0..7) = 147-dim → `Linear(147 → node_dim)`. Direct port of LigandMPNN `model_utils.py:1284-1295`.
- `atomic_number_embedding` (v2): `nn.Embedding(120, node_dim, padding_idx=0)` over atomic numbers.
- `onehot6` (v1, legacy): 6-bin element histogram (C/N/O/S/P/other).

### 3.3 Cropping policy

- Keep nearest ligand atoms up to `ligand_context_atoms=25`.
- Keep total nodes below `max_total_nodes` (curriculum: 64 → 128 → 384):
  - if too many residues, choose those nearest ligand centroid;
  - if no ligand, keep leading residues.
- Sidechain atoms are filtered and re-indexed alongside the residue crop so `sidechain_residue_idx` always points into the post-crop residue list.

### 3.4 Sidechain-as-context augmentation (v3, training-only)

Direct port of LigandMPNN's `use_side_chains` mechanism (`model_utils.py:1247-1271`):

- Pick a random `sidechain_context_rate=0.03` fraction of designable residues per sample.
- Append their sidechain heavy atoms to `ligand_coords` and `ligand_atomic_numbers` (treating them as ligand-like nodes — dense PairMixer attends to them naturally; no separate Y graph).
- Recompute `residue_ligand_frame_angles` over the augmented ligand set so the [L, M, 4] tensor stays aligned.
- Target `sequence` and `design_mask` are *unchanged* — those residues are still in the cross-entropy loss; only their sidechain geometry leaks into the context.
- Off in val. Sidechain tensors never reach `collate_batch` (consumed pre-batch by `_apply_sidechain_context_aug`).

### 3.5 Batch collation

Padded tensors:

- `residue_coords` `[B, L_max, 3]`, `residue_features` `[B, L_max, 6]`, `residue_mask` `[B, L_max]`, `sequence` `[B, L_max]`, `design_mask` `[B, L_max]`.
- `residue_backbone_coords` `[B, L_max, 4, 3]` (when `data.pair_distance_atoms` is `backbone_full*` or frame angles are on).
- `residue_ligand_frame_angles` `[B, L_max, M_max, 4]` (when `frame_relative_angles=true`).
- `ligand_coords` `[B, M_max, 3]`, `ligand_mask` `[B, M_max]`.
- One of `ligand_atomic_numbers [B, M_max]` (v2/v3) or `ligand_features [B, M_max, 6]` (v1).

Variable-M handling: `M_max` is taken over the *post-augmentation* batch, so sidechain-augmented samples can have larger M than non-augmented ones in the same batch.

## 4) Model architecture

### 4.1 Initialization

- `h_res = Linear(residue_features [B, L, 6] → node_dim)`.
- `h_lig = ligand_featurizer(ligand_atomic_numbers or ligand_features)` (per §3.2).
- `h = LayerNorm(concat(h_res, h_lig))` along the node axis.
- `pair_mask = node_mask[:, :, None] & node_mask[:, None, :]`.

### 4.2 Invariant dense pair tensor

Construct `Z [B, N, N, pair_dim]` (`N = L_res + L_lig`) from:

- `Z = pair_i(h)[:, :, None] + pair_j(h)[:, None, :] + relpos_emb(rel)`
- plus distance-based components, conditioned by `pair_distance_atoms` (L-L block) and `pair_distance_atoms_ligand` (L-M / M-L blocks):

| Flag | Block | Behavior |
|---|---|---|
| `pair_distance_atoms='anchor_only'` (legacy) | L-L | single-atom Cβ–Cβ distance, RBF expansion |
| `pair_distance_atoms='backbone_full'` (v2) | L-L | 5 backbone-pair distances (Cα-Cα, Cα-N, Cα-C, N-O, O-C), stacked → `Linear(5·num_rbf → pair_dim)` |
| `pair_distance_atoms='backbone_full_25'` (**v3 default**) | L-L | full LigandMPNN 25 backbone-pair set, stacked → `Linear(25·num_rbf → pair_dim)` |
| `pair_distance_atoms_ligand='anchor_only'` (legacy) | L-M, M-L | single-atom anchor-to-ligand distance |
| `pair_distance_atoms_ligand='backbone_full'` (**v3 default**) | L-M, M-L | 5 backbone-atom × ligand-atom distances → `Linear(5·num_rbf → pair_dim)` |

Optional concat into the L-M block:
- `frame_relative_angles=true` (**v3 default**): per-(residue, ligand-atom) `[f1, f2, f3, f4]` features computed in the residue's local N-Cα-C frame (LigandMPNN `_make_angle_features`, model_utils.py:1123-1147), projected via `Linear(4 → pair_dim)` and added to the L-M block.

Optional opt-in extras (default OFF in v3 — non-LigandMPNN, kept as future-ablation switches):
- `intra_ligand_multidist`: K-NN intra-ligand multi-distance signature added to ligand node embeddings before the pair tensor (architectural, not a pure feature add — disabled in v3 to avoid re-introducing the architectural confound).

Coordinate noise (training-only):
- `training.coord_noise_std=0.1` (**v3 default**): Gaussian σ=0.1 Å added to raw `X` (N/Cα/C/O) and raw `Y` (ligand) at the top of `forward()`. If `residue_anchor='cb'`, virtual Cβ is recomputed from the noisy X so the Cβ = f(N, Cα, C) invariant is preserved (matches LigandMPNN `model_utils.py:1189-1191`).
- Legacy `thermal_noise_std` (post-cdist pair-distance noise) remains for backwards compatibility but defaults to 0.

### 4.3 PairMixer block

Each block applies, with residual updates:

1. `Z = Z + TriangleMultiplicationOutgoing(Z, pair_mask)` — AF2-style: norm + 4 linears (a, b, gate_a, gate_b → hidden_dim) + einsum `bikd,bjkd→bijd` + output norm + output gate + linear out.
2. `Z = Z + TriangleMultiplicationIncoming(Z, pair_mask)` — same with `bkid,bkjd→bijd`.
3. `Z = Z + Transition(Z)` — LN + Linear(↑`pair_transition_mult`×) + GELU + Linear(↓1×) + Dropout.

Pair mask is applied after each sub-update. Gradient checkpointing is enabled for stage 3 (and optional elsewhere) — controlled by `model.gradient_checkpointing`.

Default scale (held constant across v2 → v3):
- `node_dim = pair_dim = pair_hidden_dim = 128`
- `num_pairmixer_blocks = 6`
- `pair_transition_mult = 4`
- `num_rbf = 32`, `max_distance = 24 Å`
- ~2.31 M trainable parameters; 86 % live in the encoder.

### 4.4 Decoder

Three contexts feed the per-residue head:

1. `node_repr_res` — node_dim vector. Built by an attention-pooled readout over the pair-row: per-residue learned weights over all node columns (`pair_readout_attn`), then a linear projection summed into the original node features.
2. `ar_context` — node_dim vector. Multi-head causal attention where the *attention logits* are derived from the pair tensor (`ar_pair_to_attn: pair_dim → ar_num_heads`) and the *values* come from token embeddings of previously-decoded residues. Decoding order is supplied per-batch (default left-to-right). Causal mask blocks future positions; `residue_mask` zeros out padding.
3. `ligand_context` — pair_dim vector. Residue-to-residue + residue-to-ligand attention pooling over `Z`, projected to pair_dim via `ctx_proj`.

Final readout: `LN(concat(node_repr_res, ar_context, ligand_context))` → `Linear(2·node_dim + pair_dim → node_dim)` → GELU → Dropout → `Linear(node_dim → 21)`.

The decoder is intentionally thin (single hidden layer); deepening is a v4 candidate.

## 5) Loss and optimization

### 5.1 Loss

Cross-entropy on designed and valid positions:

- Valid positions = `residue_mask & design_mask`.
- Padded or fixed positions ignored via target `-100`.

### 5.2 Metrics

- `train/loss`, `val/loss`
- `train/acc`, `val/acc` over valid positions
- `train/lr` (logged via Lightning)

### 5.3 Optimizer + schedule

- AdamW, `lr=3e-4`, `weight_decay=1e-2`.
- Linear warmup over `training.warmup_steps`, then cosine decay over `training.T_max`.
- `gradient_clip_val=1.0`.

## 6) Curriculum + DDP profile

3-stage curriculum (v3, identical schedule to v2 for direct comparability):

| Stage | GPUs | `max_total_nodes` | bsz/rank | epochs | wall-time est. |
|---|---|---|---|---|---|
| 1 | 1×A5500 | 64 | 8 | 15 | ~12–14 h |
| 2 | 4×A5500 DDP | 128 | 4 | 25 | ~12–14 h |
| 3 | 8×A5500 DDP | 384 | 2 | 30 | ~11–21 days |

Stage transitions use weights-only `init_from_checkpoint` (parameter shape diffs are reported in the log). Canonical v3 ckpt = stage-3 epoch with min `val/loss`.

Runtime profile (default):
- precision: `bf16-mixed`
- gradient clipping: 1.0
- gradient checkpointing: ON for stage 3, OFF earlier
- pin_memory: true
- num_workers: 8

## 7) Training and inference interfaces

### 7.1 Preprocess

```bash
sbatch scripts/SLURM/01b_preprocess.sh    # --recompute by default
```

Emits a *union cache*: every `.pt` carries every v2/v3 tensor key (both featurizer outputs, backbone coords, frame angles, sidechain atoms). Runtime config selects which subset to consume.

### 7.2 Train

```bash
J1=$(sbatch --parsable scripts/SLURM/04a_v3_train_stage1.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/SLURM/04b_v3_train_stage2_ddp.sh)
J3=$(sbatch --parsable --dependency=afterok:$J2 scripts/SLURM/04c_v3_train_stage3_ddp.sh)
```

Hydra `++key=value` overrides are required (config defaults are v2; v3 SLURMs override the v3 flag bundle explicitly).

### 7.3 Pilot (smoke test before curriculum)

```bash
sbatch scripts/SLURM/02_pilot_v3.sh    # 1×A5500, all v3 flags ON, 1-batch overfit
```

### 7.4 Inference

```bash
uv run python scripts/inference.py --pdb <path> --ckpt <ckpt> --out_fasta <path>
```

Workflow: parse and featurize, forward pass, argmax or temperature sample, write FASTA. Supports per-residue chain-letter selection for fixed/redesign masks (e.g., `A23,B42`).

## 8) Validation checklist

Before extending architecture or launching multi-week training:

1. `make lint` — ruff clean.
2. `make test` — pytest passes. CPU-only, ~75 s for 184 tests.
3. `test_v3_off_equals_v2_baseline` regression guard passes (v3 codebase with all v3 flags off must produce numerically identical logits to v2 on a frozen batch).
4. `sbatch scripts/SLURM/02_pilot_v3.sh` — pilot run shows `train/loss` decreasing monotonically, `val/loss` finite, no NaN.
5. Stage 1 final `val/acc` ≥ v2 stage-1 baseline (0.426). Notable regression → integration bug, halt.

## 9) Feature flag surface

All flags live under `data.*` (authoritative) and mirror to `model.*` via Hydra interpolation.

| Flag | v1 | v2 | v3 |
|---|---|---|---|
| `data.ligand_featurizer` | onehot6 | atomic_number_embedding | **ligandmpnn_atomic** |
| `data.residue_anchor` | ca | **cb** | **cb** |
| `data.pair_distance_atoms` | anchor_only | backbone_full | **backbone_full_25** |
| `data.pair_distance_atoms_ligand` | anchor_only | anchor_only | **backbone_full** |
| `data.frame_relative_angles` | false | false | **true** |
| `data.return_sidechain_atoms` | false | false | **true** |
| `data.sidechain_context_rate` | 0.0 | 0.0 | **0.03** |
| `training.coord_noise_std` | 0.0 | 0.0 | **0.1** |
| `model.intra_ligand_multidist` | false | false | false (opt-in extra; not in v3 baseline) |
| `model.thermal_noise_std` | 0.0 | 0.0 | 0.0 (legacy; superseded by `coord_noise_std`) |
| `model.gradient_checkpointing` | false | per-stage | per-stage |

## 10) v3 → v4 candidates

Architecture knobs to revisit *after* the v3 vs LigandMPNN comparison is published:

- **Encoder depth**: 6 → 8–12 PairMixer blocks. Highest-ROI quality lever; gradient checkpointing already keeps memory in budget at stage 3.
- **Decoder depth**: 1 → 2–3 hidden layers. Currently disproportionately thin (2.3 % of params).
- **Sidechain coordinate auxiliary head**: add a coord-loss term — predict χ angles or sidechain heavy-atom positions alongside the sequence head.
- **Random decoding order**: replace fixed left-to-right with sampled permutations during training, à la ProteinMPNN.
- **Distance-noise scheduling**: anneal `coord_noise_std` over training rather than holding constant.
- **Diffusion-style ligand-aware refinement**: optional iterative head conditioning sequence on partial design.

These are out of scope for v3 — v3's value depends on holding architecture constant relative to v2 so the feature deltas are the only moving variable.
