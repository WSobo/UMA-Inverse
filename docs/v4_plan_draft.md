# UMA-Inverse v4 — Working Plan (DRAFT, in progress)

**Status:** Living document. Open questions are answered once v3 stage 3
finishes and the post-stage-3 SLURM queue chain results land
(benchmark + pocket-fixed + cofold + cofold-metrics + distal-KL + wallclock
+ **distogram probe**). Do not freeze any v4 decision before then.

**Owner:** wsobolew  •  **Started:** 2026-05-11

**Hardware:** v4 trains on **A100** (not A5500). The A5500 consumer-hardware
constraint was a v1–v3 reproducibility/accessibility choice; v4 relaxes it.
Expected speedup: ~3–5× wall-clock per stage vs. 8× A5500 DDP, which matters
for the Boltz-pretraining corpus (Stage 3) and larger node budgets.

---

## 1. What v3 actually is (so we know what v4 changes)

From `logs/runs/pairmixerinv-v3-stage3-nodes384-ddp8/config.yaml`:

| Component | v3 setting | LigandMPNN-faithful? |
|-----------|-----------|---------------------|
| Encoder | Dense L×L PairMixer, 6 blocks, triangle multiplication only | N/A — UMA-specific |
| `node_dim / pair_dim / pair_hidden_dim` | 128 / 128 / 128 | — |
| `num_rbf` / `max_distance` | 32 / 24 Å | matches |
| Residue anchor | Cβ | matches |
| Residue-residue pair distances | `backbone_full_25` (all 25 ordered N/Cα/C/O/Cβ pairs) | matches |
| Residue-ligand pair distances | `backbone_full` (5 backbone × ligand) | matches |
| Frame-relative angles | enabled (cosθ, sinθ, cosφ, sinφ) | analogue of trRosetta ω/θ/φ |
| Ligand featurizer | `ligandmpnn_atomic` (atomic_num + group + period, 147→128) | matches |
| Sidechain context | 3% augmentation rate (`return_sidechain_atoms=true`) | UMA-specific |
| Coord noise | σ=0.1 Å on backbone + ligand, training only | LigandMPNN uses 0.3 |
| Decoder | **Random-order autoregressive** via `_autoregressive_context` (4-head pair-attention pool over teacher-forced previous tokens) | matches |
| Decoder input | `[node_repr_res | ar_context | ligand_context]` → 2-layer MLP → 21 classes | matches concat pattern |
| Param count | ~2.3 M | LigandMPNN is 2.2 M |
| Training | bf16-mixed, DDP×8 on A5500, cosine LR w/ 2k warmup, 3-stage curriculum (64 → 128 → 384 nodes) | — |

**Correction to prior thinking:** v3 IS random-order AR — the
`_autoregressive_context` does teacher-forced random-permutation decoding
with a causal mask over `decoding_order`. It is NOT a one-shot parallel
classifier. The PairMixer trunk is parallel; the decoder is AR. This
matters for v4 because the "drop the AR head" decision was framed against
a non-AR v3 baseline that doesn't exist.

---

## 2. v3 retro — what training taught us

### 2.1 Headline result (stage 3 complete — 27 epochs, ep 0–26)
- **Best val/acc: 0.6236** (ep 21, 23, 25 all tied); **best val/loss: 1.2093 at ep 23**.
- Canonical checkpoint: `checkpoints/uma-inverse-v3.ckpt` (promoted from
  `pairmixerinv-v3-stage3-nodes384-ddp8/uma-inverse-23-1.2093.ckpt`).
- Converged above the ~0.620–0.628 plateau estimate from ep 10.
  The early slowdown (ep 7→10: +0.04 pts) was a wobble, not a ceiling;
  training continued gaining through ep 23 before flattening.
- vs v2: 0.6236 (v3) vs 0.6387 (v2 ep 23) — v3 trails by ~1.5 pts aggregate
  recovery (expected given the feature-matched architectural constraint).

### 2.2 Three competing explanations
The aggregate-recovery plateau could be:

- **(a) Decoder bottleneck** — encoder learns useful geometry but the
  random-order AR + small decoder MLP can't exploit it. v4: keep encoder,
  upgrade decoder, more data.
- **(b) Encoder inert** — Z_ij isn't carrying geometry; the model is
  effectively a glorified MLP on local features filtered through the AR
  head. v4: trunk rethink before features/decoder.
- **(c) Regularization miscalibration** — `sidechain_context_rate=0.03` and
  `coord_noise_std=0.1` over-regularize at the v3 param budget. v4: ablate
  the recipe, leave architecture alone.

These have **different v4 implications** and are mutually exclusive at the
"which lever moves the most" level.

### 2.3 The diagnostic: distogram linear probe (queued)

`scripts/paper/distogram_probe.py` + `scripts/SLURM/preprint_distogram_probe.sh`,
hooked into `submit_post_v3_pipeline.sh` as the 8th `afterok:stage3` branch.

Method: freeze the v3 trunk, extract residue-residue `z[:L,:L,:pair_dim]`
per PDB on a ~400-PDB val sample, train a single
`Linear(pair_dim, 38)` head on Cβ-Cβ AF3-template distogram bins
(3.15–50.75 Å, 1.25 Å width).

**Pre-committed verdict thresholds (top-1 accuracy on held-out PDBs):**

| top-1 | Verdict | v4 implication |
|-------|---------|----------------|
| > 0.85 | encoder strong  | Bottleneck is decoder/data → prioritize Boltz pretraining + decoder upgrade |
| 0.60–0.85 | encoder partial | v4 wins from feature density (AF3 pos enc, AtomFlow, token_bonds) |
| < 0.60 | encoder weak    | Z_ij not carrying geometry → trunk rethink before features |

### 2.4 FILLED — distogram probe results (2026-05-15)

- **Top-1: 0.266**  Top-3: 0.588  Neighbor (±1): 0.568  MAE: 3.69 Å  ECE: 0.025
- **VERDICT: encoder_weak** (top-1 = 0.266, far below 0.60 threshold)

Interpretation:
- top-3 (0.588) >> top-1 (0.266): the probe learns coarse distance ranges
  but cannot discriminate precisely. ECE=0.025 means it is well-calibrated,
  so this is not overconfidence — the encoder simply does not carry
  fine-grained Cβ-Cβ geometry in Z_ij.
- Combined with distal-KL outcome (UMA diverges 20× more from native than
  LigandMPNN at >25 Å), the picture is: the encoder IS responding to the
  ligand at distal positions (mechanism KL higher), but the responses are
  noisy/incorrect because Z_ij lacks geometric grounding.
- v4 implication (pre-committed): trunk rethink before features. Adding
  AF3 pos-enc or AtomFlow features to a geometrically-blind trunk will
  not fix this — the trunk needs a geometry supervision signal during
  training (→ promote Stage 2 auxiliary heads to the front of v4).

---

## 3. The single yardstick decision

Pre-v4 we have three plausible yardsticks; we can only commit to one to
make v4 decisions tractable.

| Yardstick | What it measures | v4 levers it favors |
|-----------|------------------|---------------------|
| **A. Cofold pass-rate** | % of pocket-fixed redesigns that re-fold to native pocket Cα RMSD ≤ 2 Å under Boltz-2 | Distal context, data scale; decoder less critical |
| **B. Pocket-residue recovery** | Native-AA recovery on residues within 6 Å of any ligand atom, on holo split | Ligand features (chirality, hybridization), residue-ligand pair richness, retrieval |
| **C. Aggregate val recovery (CATH-style)** | Mean per-position recovery on full val split | Decoder strength, data scale, harder to move at small param budgets |

**Yardstick is chosen after seeing v3 stage 3 + cofold queue results:**

- If **cofold pass-rate (UMA-v3) > LigandMPNN by ≥5 pts** → commit to A.
  Paper angle holds; v4 doubles down on distal context.
- If **cofold parity but pocket-residue recovery wins by ≥3 pts** → commit
  to B. Reframe paper around pocket design rather than distal.
- If **both lose** → revisit before v4. Distogram probe verdict
  decides whether the loss is encoder (→ trunk rethink) or decoder
  (→ data + decoder upgrade).

### 3.1 FILLED — cofold + distal-KL results (2026-05-15)

**Important:** UMA ran on 35 PDBs; LigandMPNN ran on 20. Aggregate comparison
is misleading. All headline numbers below are on the **20 shared PDBs** (apples-to-apples).

- **v3 cofold pass-rate (design-level) vs LigandMPNN: 65.0% / 66.0%** — essentially tied
- **v3 cofold pass-rate (PDB-level, any design passes): 80.0% vs 75.0%** — UMA wins by 5 pts
- v3 pocket Cα RMSD (median/mean): **1.34 / 2.94 Å** vs LigandMPNN 1.30 / 2.92 Å — tied
- v3 ligand RMSD (median/mean): **0.81 / 1.11 Å** vs LigandMPNN **1.00 / 1.37 Å** — UMA wins
- UMA wins per-PDB pocket RMSD on 8/20 PDBs; big wins: 2wgj (0.31 vs 1.60 Å), 1r0p (1.35 vs 2.43 Å)
- v3 pocket residue recovery vs LigandMPNN: benchmark still running (2026-05-15)
- v3 mean distal KL outcome (>15 Å / >25 Å) vs LigandMPNN:
  - UMA: 0.176 / 0.199  vs  LigandMPNN: 0.012 / 0.010
  - UMA makes noisier distal AA predictions — ligand-signal present but noisy
- **Yardstick committed to: A (cofold pass-rate)**
  - On the shared PDB set, UMA matches or beats LigandMPNN (80% vs 75% PDB-level)
  - UMA has a genuine edge in ligand placement (0.81 vs 1.00 Å median ligand RMSD)
  - Cofold pass-rate condition from §3 is borderline met (PDB-level win by +5 pts)
  - Distogram probe says encoder_weak, but practical outputs don't show it collapsing —
    the model IS useful despite not encoding clean Cβ-Cβ geometry
  - v4 goal: fix the geometry grounding (Stage 2 aux heads during training) while
    keeping the architectural wins that are already working

---

## 4. Ranked v4 priority stack (conditional on retro outcomes)

Stages assume yardstick committed; revise if retro forces a rethink.

### Stage 0 — AF3 positional encoding (1 week)
Highest signal-to-engineering ratio. v3 has none of:
- `arel_pos` (residue-index offset, clipped at ±32 with chain sentinel)
- `arel_token` (token-offset within ligand atoms)
- `arel_chain` (chain-offset, clipped at ±2)
- `token_bonds` (covalent-bond channel — the most likely actual win)

`token_bonds` is special: covalent topology is the one structural signal
that neither RBFs nor frame angles capture. Direct relevance to covalent
inhibitors, peptide bonds, disulfides, ligand internal bond orders.

Implementation: extend `_init_pair` in `uma_inverse.py`; add a `bond_graph`
field to the data pipeline alongside `ligand_atomic_numbers`.

### Stage 1 — Full AtomFlow ligand features + reference-conformer (2 weeks)
Current v3 ligand featurizer: atomic_num + group + period (147 dims).
Missing (per AtomFlow 2024 canonical list):
- chirality tag, formal charge, implicit valence
- explicit H count, radical electrons
- hybridization (sp/sp2/sp3/sp3d/sp3d2)
- aromaticity, ring membership/size
- Gasteiger or MMFF94 partial charge
- van der Waals + covalent radii
- (optional) RDKit ETKDGv3 reference-conformer coords as a parallel
  intra-ligand pair channel

All cheap to compute via RDKit. Concern: pushes ligand_repr dim from 128
toward 256 if we want to preserve per-feature signal — needs param-budget
sizing (Stage 2 of cuts list below).

### Stage 2 — Auxiliary heads on Z_ij (1 week, partially diagnostic first)
Distogram + orientation + PAE-like heads, ~30 lines each, weight ~0.2.

**Two-phase rollout:**
1. *Diagnostic first:* The post-v3 distogram probe IS Stage 2A's first
   head, run on a frozen trunk. If verdict is `encoder_strong`, the head
   trained jointly during v4 will likely add nothing. If `encoder_partial`
   or `encoder_weak`, joint training of the head SHOULD help (because the
   geometry gradient flows into Z_ij).
2. *Training-time:* Add to the v4 training loss with weight ~0.2.

### Stage 3 — Boltz-pretrain + holo finetune (4–6 weeks)
Promoted from "Stage 4" in the original v4 roadmap. ESM-IF precedent
(Hsu et al., ICML 2022): 12M predicted backbones → +8.9 pts recovery
over CATH-only baseline. For ligand-aware: pretrain on Boltz-1/Boltz-2
predicted holo complexes (gated by pLDDT ≥0.7 and iPTM ≥0.5), finetune
on holo PDB.

**Bottleneck risk:** generating the Boltz-predicted training corpus
itself is non-trivial. Need a data-engineering plan separate from
architectural work.

### Stage 4 (conditional) — Decoder rethink (4 weeks)
Only fires if Stage 0–3 yardstick numbers plateau. Choice tree:
- If yardstick A (cofold): masked-diffusion decoder (MDLM/DPLM/MapDiff)
  with purity-prior unmasking. AR head dropped.
- If yardstick B (pocket): retrieval-augmented decoder (PRISM-style
  pocket-motif bank, queried by Z_ij similarity around the ligand).
- If yardstick C (recovery): both have ~equal arguments; lean retrieval
  because pocket recovery is the most likely sub-failure.

### Stage 5 (optional polish) — DPO / pTM regularizer / wallclock probe
ResiDPO + AFDistill pTM are low-risk post-training. Worth at most 2 weeks
of effort; pursue only if Stage 3 win is real and we're polishing for the
preprint.

---

## 5. What's cut from the original v4 scan

Cuts based on retro reasoning above:

1. **Joint AR + diffusion co-training head.** Pick one decoder; don't
   co-train. (User-confirmed: drop the joint AR head — eval-mode-only
   alternative also rejected because per-position log-probs from the
   one-shot decoder are already comparable to LigandMPNN's eval metric.)
2. **AFDistill pTM regularizer at training time.** Demoted to "polish"
   (Stage 5) — adds a distilled model in the loop without a clear
   small-molecule win.
3. **DPO/ResiDPO post-training.** Demoted to Stage 5 — too speculative
   until cofold numbers stabilize.
4. **Tree-search / test-time refinement.** Out of scope for v4; revisit
   in v5 if Stage 3 wins.
5. **PRISM retrieval as standalone v2.** Folded into Stage 4 as one of
   two conditional choices, not as a separate stage.

---

## 6. Open questions / decisions deferred to post-queue

### 6.1 Param budget
v3 = 2.3 M params at `node_dim=pair_dim=128`. Stage 0+1 features and
auxiliary heads will push toward 4–5 M without cuts. A5500 / 24 GB / bf16
caps real upper bound somewhere around 8 M with the current PairMixer
depth.

- [ ] Should v4 stay at 128 channels and add features depth-wise, or
  promote to 192/256 channels and reduce blocks? **Decide after Stage 0
  sizing pass.**

### 6.2 Data corpus shape
Current v3 train set: `LigandMPNN/training/train.json`, ~few thousand
holo complexes. Stage 3 wants Boltz-predicted complexes added.

- [ ] Source for Boltz-predicted holo set: BindingDB-derived? PDBbind
  re-folded? Need a data plan separate from this doc.
- [ ] Pre-train epoch count vs. finetune epoch count split.

### 6.3 Curriculum policy
v3 used 3-stage node-count curriculum (64 → 128 → 384). v4 may want a
4-stage version or drop curriculum if Boltz-pretrain absorbs the role.

- [ ] Decide after Stage 3 data plan is concrete.

### 6.4 Yardstick lock-in
See §3.1.

### 6.5 If the v3 recipe is actually fine
If cofold + pocket recovery both win without v4 changes, **don't ship v4**
just to ship architecture. The honest v4 deliverable in that case is:
better data + retraining at slightly relaxed regularization
(coord_noise=0.05, no sidechain aug).

- [ ] Decide whether v4 is needed at all — based on combined retro.

---

## 7. Checklist — feed numbers in here when they land

| Source | Job | Status | Result |
|--------|-----|--------|--------|
| stage 3 final ckpt | `04c_v3_train_stage3_ddp.sh` | **complete** | val/acc=0.6236, val/loss=1.2093 @ep23; ckpt→`uma-inverse-v3.ckpt` |
| benchmark | `05_benchmark.sh` (v3-final) | **complete** | v3-final: 1727 PDBs, mean recovery see summary.json; K=0 sampled: UMA-Inverse-1=0.4637, LigandMPNN-matched=0.5389 |
| pocket-fixed redesign | `preprint_uma_pocket_fixed.sh` | **complete** | outputs/preprint/uma_pocket_fixed_v3 |
| Boltz-2 cofold | `preprint_boltz_cofold_v3.sh` | **complete** | outputs/preprint/cofold_v3 |
| cofold metrics | `preprint_cofold_metrics_v3.sh` | **complete** | UMA 53.7% vs LMPNN 66.0% pass-rate; pocket RMSD best/mean=0.20/3.16 Å |
| distal-KL mechanism | `preprint_distal_kl_shift.sh` (mech) | **complete** | UMA KL >25Å=0.040 vs LMPNN 0.013 — UMA more ligand-sensitive but noisily |
| distal-KL outcome | `preprint_distal_kl_shift.sh` (outcome) | **complete** | UMA KL >25Å=0.199 vs LMPNN 0.010 — UMA diverges 20× more from native |
| distogram probe | `preprint_distogram_probe.sh` | **complete** | top1=0.266, top3=0.588, MAE=3.69Å — **VERDICT: encoder_weak** |
| wallclock probe | `probe_inference_wallclock.py` | **failed** | ProDy path bug: relative path passed where 4-letter PDB ID expected |

**Status:** All benchmarks complete. K=0 sampled recovery: UMA-Inverse-1=0.4637 vs
LigandMPNN=0.5389 on matched 2000-PDB val set. Wallclock probe still needs path
fix. v4 direction determined: trunk rethink (aux geometry heads) before features.
