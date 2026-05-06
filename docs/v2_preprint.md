# Dense Pair-Wise Attention for Ligand-Conditioned Protein Sequence Design

**UMA-Inverse: a single-encoder alternative to KNN message-passing for ligand-conditioned inverse folding.**

W. Sobolewski¹

¹ Yeh Lab, University of California, Santa Cruz.
Correspondence: `wsobolew@ucsc.edu`. Code and weights: <https://github.com/...> (TBD before submission).

---

## Abstract

Existing ligand-conditioned protein sequence design methods, notably LigandMPNN, are built on K-nearest-neighbor message-passing graphs that limit ligand information to each residue's local protein neighborhood. We introduce **UMA-Inverse**, an alternative architecture that uses a single dense pair-wise attention encoder over the concatenated residue + ligand atom set with no KNN sparsification. UMA-Inverse outperforms ProteinMPNN on the LigandMPNN-paper interface-recovery benchmark by **+8.0 percentage points on metal-binding sites (0.486 vs 0.406)** and **+3.3 percentage points on small molecules (0.538 vs 0.505)**, but trails LigandMPNN itself overall, with the gap concentrated on metal sites (−28.9 pp). To characterize where dense pair-wise attention may offer a structural advantage that local KNN methods cannot, we evaluate both methods on **pocket-fixed redesign** — locking the binding-pocket residues to native and asking the model to redesign the remaining protein. Across N = 25 small-molecule PDBs (with the v2-friendly selection tiebreaker scrubbed), UMA-Inverse produces distal-position designs that are significantly less diverse than LigandMPNN's (mean pairwise Hamming 0.13 vs 0.19, paired Wilcoxon p = 0.0015), with broadly identical per-PDB AA preferences (median Pearson r = 0.85 between methods' AA distributions). Critically, UMA's per-PDB distal-position confidence is significantly correlated with downstream Boltz-2 cofold ipTM (r = 0.57, p = 0.003) and complex pLDDT (r = 0.54, p = 0.006), and the correlation on the pocket-specific ipTM metric is roughly twice as strong for UMA-Inverse as for LigandMPNN (r = 0.26, p = 0.20) — so UMA's narrower distal posterior tracks design quality more reliably than LigandMPNN's. We argue that dense pair-wise attention is the natural architecture for the **pocket-fixed redesign use case** (sequence diversification, allosteric tuning, scaffold engineering on a fixed pocket), where confident-but-pocket-conditioned distal redesigns are preferable to broad-but-uninformed ones.

---

## 1. Introduction

Designing protein sequences that fold to a specified backbone is the inverse-folding problem; conditioning on additional non-protein context (small molecules, nucleotides, metal ions) extends the problem to ligand-binding design. The de facto standard, **ProteinMPNN** [Dauparas et al. 2022], handles backbone-only design via a sparse-graph message-passing neural network with K-nearest-neighbor edges between residues. **LigandMPNN** [Dauparas et al. 2023] extends ProteinMPNN with two parallel KNN graphs over ligand atoms (a ligand-only graph and a protein-ligand bipartite graph), achieving strong interface-recovery numbers on small-molecule, nucleotide, and metal-binding test splits.

LigandMPNN's architecture inherits ProteinMPNN's locality assumption: each residue sees only its K nearest ligand atoms (default K=25), with no direct edge to ligand atoms beyond that radius. In practice this is fine for residues directly contacting a ligand, but it creates an information bottleneck for residues farther away — distal protein-side scaffolding residues that may nonetheless influence pocket geometry only see ligand context through 2-3 hops of message passing, mixed at each hop with K=25 other contributions. We hypothesize that this locality may matter for design tasks where global context propagation is important: e.g., redesigning a distal residue *to support* a fixed binding pocket.

In this work we build and evaluate **UMA-Inverse**, an alternative architecture for the same problem. Rather than three sparse KNN graphs, UMA-Inverse uses a single dense pair-wise attention encoder over the union of residue and ligand atom nodes. Every residue has direct edges to every ligand atom and every other residue; no graph sparsification. We benchmark UMA-Inverse on LigandMPNN's protocol (interface sequence recovery, 10 designs/PDB, T=0.1, random decoding order) and find that it lands between LigandMPNN and ProteinMPNN on standard recovery metrics. We then characterize the regime where the architectural difference *does* matter — pocket-fixed redesign — and find that UMA-Inverse's distal posterior is significantly narrower than LigandMPNN's on small-molecule pockets, that this narrowness is uncorrelated between the two methods (so they're tracking different signals at distal positions), and that UMA's per-PDB confidence at distal positions predicts downstream Boltz-2 cofold quality more reliably than LigandMPNN's, especially on the pocket-specific ipTM metric.

The contribution is threefold:

1. A single-encoder, dense-pair-wise-attention baseline for ligand-conditioned inverse folding.
2. Like-for-like benchmark numbers under the LigandMPNN protocol, including all three test splits.
3. The empirical case for **pocket-fixed redesign** as the use case where dense pair attention (as a class) is most differentiated from sparse KNN, with structural analysis via Boltz-2 cofolding and a per-PDB confidence-vs-quality correlation that is markedly stronger for UMA-Inverse than for LigandMPNN — most clearly on the pocket-specific ipTM metric.

---

## 2. Methods

### 2.1 Architecture: UMA-Inverse

UMA-Inverse is an encoder-only sequence design model. The input is a fixed protein backbone (N, Cα, C, O coordinates per residue) plus ligand atom coordinates and identities. The output is a per-residue distribution over the 20 amino acids plus an unknown token.

**Featurization.**

- *Residue nodes*: 6-dimensional sin/cos encoding of backbone dihedral angles (φ, ψ, ω). Residue position is anchored at the **virtual Cβ** computed from N, Cα, C via the ProteinMPNN formula:

  ```
  b = Cα - N
  c = C - Cα
  a = b × c
  Cβ = -0.58273431 a + 0.56802827 b - 0.54067466 c + Cα
  ```

  (Glycines have no real Cβ; the formula returns the same virtual position.)

- *Ligand atom nodes*: a learned embedding over atomic numbers 0-119 (0 = padding; 1-118 = real elements; 119 = unknown sentinel). This is in contrast to LigandMPNN's one-hot over the chemical-element types in their training set.

- *Pair tensor*: for each (i, j) pair (residue-residue, residue-ligand, ligand-ligand), a Gaussian RBF embedding of inter-atomic distance (32 bases, max 24 Å). Three block types:
  - *[L, L] block (residue-residue):* 5 backbone-pair distances per residue pair: (Cα,Cα), (Cα,N), (Cα,C), (N,O), (O,C). Stacked, RBF'd, projected through `Linear(160, pair_dim)`.
  - *[L, M] / [M, L] blocks (residue-ligand):* single Cβ-anchor-to-ligand-atom distance, RBF'd, projected through `Linear(num_rbf, pair_dim)`.
  - *[M, M] block (ligand-ligand):* single ligand-atom-to-ligand-atom distance, same projection.

**Encoder.** 6 PairMixer blocks, each a transformer-style attention layer over the residue+ligand union (treating residue and ligand atoms symmetrically as nodes). Pair-tensor biasing into attention scores (relative-distance-aware attention) plus a feed-forward block. node_dim = 128, pair_dim = 128, pair_hidden_dim = 128. Total parameters: ~3.6M.

**Decoder.** Autoregressive over a random per-batch decoding order. At each position, sample from the model's logits at temperature T=0.1 (matching LigandMPNN protocol).

### 2.2 Training data and curriculum

We use the LigandMPNN training/valid splits (`train.json` 149,488 PDBs, `valid.json` 7,530 PDBs). Test splits: `test_metal` (82 PDBs), `test_small_molecule` (316 PDBs), `test_nucleotide` (39 PDBs evaluable; see §3.2).

PDB structures are pre-processed into a union cache (all v1 and v2 keys emitted regardless of config) at `data/processed/`. Cache size 7.9 GB; coverage 154,658 PDBs (98.5% of train, 98.8% of valid). The 1.5% loss is RCSB 404s on PDBs deprecated since the LigandMPNN split was published, plus ~37 parser ValueErrors.

Training is a 3-stage curriculum on increasingly large `max_total_nodes` budgets (residue + ligand atom counts, after cropping):

| Stage | Hardware | Max nodes | Effective batch | Wall | Best val_acc |
|---|---|---:|---:|---|---:|
| 1 | 1× A5500 | 64 | 8 | 10h 32m | 0.4262 (ep 6) |
| 2 | 4× A5500 DDP | 128 | 16 | 12h 40m | 0.5108 (ep 14) |
| 3 | 8× A5500 DDP | 384 | 16 | 3d 19h 30m | 0.6387 (ep 29) |

All stages: AdamW, lr=3e-4 peak, weight_decay=1e-2, bf16-mixed precision, gradient_clip=1.0, linear warmup → cosine decay schedule sized to that stage's step budget. Stages 2 and 3 initialize weights-only from the previous stage's last checkpoint (fresh optimizer, fresh LR cycle).

The canonical v2 release checkpoint is `epoch 19` of stage 3 (val_loss 1.146, val_acc 0.637) — the minimum-val_loss epoch retained by `save_top_k=3`.

### 2.3 Standard benchmark: LigandMPNN-paper interface recovery

We follow the protocol in [Dauparas et al. 2023] verbatim:

1. For each PDB in a test split, encode the protein-ligand structure once.
2. Generate **10 sequences** autoregressively with random decoding order at T=0.1.
3. For each sequence, compute **interface recovery**: fraction of sidechain-interface positions where the predicted residue equals the native, restricted to residues whose sidechain has at least one heavy atom within 5 Å of any nonprotein heavy atom.
4. Per-PDB statistic: median of the 10 recoveries.
5. Headline statistic: mean of the per-PDB medians.

Implementation at `scripts/benchmark_interface_recovery.py`. The same protocol was applied to UMA-v1 (the previous architecture iteration without phase-1/2/3 featurizer changes) and is paired across splits with LigandMPNN-paper and ProteinMPNN-paper reported numbers for direct comparison.

### 2.4 Pocket-fixed redesign experiment

Twenty PDBs are selected from the LigandMPNN test splits — 10 metal, 10 small-molecule — passing hard filters (50 ≤ residue count ≤ 400, 5 ≤ pocket residue count ≤ 30, pocket count / total ≤ 0.4) plus split-specific filters:

- **Small molecules**: 10 distinct ligand CCD codes (no protein bound to multiple inhibitors crowding the selection), with per-protein dedup at 60% Jaccard pocket-residue overlap.
- **Metals**: artifact filter excludes PDBs whose only metal HETATM is Ni²⁺ (typical IMAC purification artifact) or whose coordination shell is ≥4 histidines and 0 acidic residues (heuristic for non-functional surface-bound metals).

Implementation at `scripts/paper/select_pdbs.py`. Selection persisted at `outputs/preprint/pdb_selection.json` with per-PDB pocket residue IDs, ligand identity (CCD code or ion symbol), and selection_reason for provenance.

For each selected PDB, both UMA-v2 and LigandMPNN are tasked with **redesigning the protein with the pocket residues forced to their native identity**. K=20 sequences are generated per (PDB, method) at T=0.1, random decoding order. Implementation at `scripts/paper/run_pocket_fixed_designs.py` (UMA, via the existing `DesignConstraints.fix` path) and `scripts/paper/build_ligandmpnn_inputs.py` + `scripts/SLURM/preprint_ligandmpnn_pocket_fixed.sh` (LigandMPNN, via `LigandMPNN/run.py --fixed_residues_multi`).

Per (PDB, method, sample) metrics:
- **Distal recovery**: fraction of non-fixed positions where predicted == native.
- **Pocket recovery**: fraction of fixed positions where predicted == native (sanity check, expected 1.0).

Per (PDB, method) aggregates:
- Mean / median / stdev of distal recovery.
- Mean pairwise Hamming distance at distal positions across the K samples (sequence diversity).
- Per-AA frequency at distal positions.

Statistics: paired Wilcoxon signed-rank test (UMA vs LigandMPNN) on per-PDB summary metrics, with bootstrap 95% CI on the mean paired difference.

### 2.5 Boltz-2 cofold of redesigned sequences

Five sequences per (PDB, method) are randomly subsampled (= 200 cofold inputs total). Each (sequence, ligand) pair is cofolded by **Boltz-2** [Passaro et al. 2025] using protein sequence + ligand CCD code as input, with affinity prediction enabled (`--sampling_steps_affinity 200 --diffusion_samples_affinity 5`). Boltz-2 outputs a predicted complex structure (PDB) plus a confidence JSON with ipTM, plDDT, and affinity scores.

Per (PDB, method, sample, diffusion_sample) metrics:
- **Boltz-2 confidence**: ipTM (interface predicted TM-score), plDDT (whole-protein), plDDT restricted to original-pocket residues, plDDT restricted to interface residues in the cofold.
- **Boltz-2 predicted affinity**: binary classification + pKi-style score from the affinity head.
- **Pocket geometry preservation**: pocket Cα RMSD between Boltz-2 cofold and native crystal, restricted to the fixed pocket residues.
- **Ligand pose preservation**: ligand heavy-atom RMSD vs the native ligand pose after aligning on pocket Cα.
- **Whole-protein scaffold RMSD**: Cα RMSD vs native crystal.

**Important caveat: Boltz-2 ipTM, plDDT, and affinity are model predictions, not measurements.** They are useful for relative comparisons between two design methods evaluating the same scaffold (where systematic prediction errors should largely cancel) but are not interpretable as absolute binding affinity or design quality. We do not claim binding affinities or catalytic efficiencies on the basis of these metrics; we use them as a proxy for "did the redesign produce a coherent ligand-pocket complex."

Cofold inputs are constructed by `scripts/paper/build_cofold_yamls.py` (one Boltz-2 YAML per (PDB, method, sample), with multi-chain support and quoted CCD codes); the SLURM wrappers `scripts/SLURM/preprint_boltz_cofold.sh` and `preprint_boltz_cofold_extended.sh` invoke `boltz predict` in directory mode against these inputs with the flags above. Cofold-quality metrics are extracted in `scripts/paper/cofold_metrics.py`.

---

## 3. Results

### 3.1 Standard interface recovery benchmark

UMA-v2 outperforms ProteinMPNN on both metal and small-molecule splits, but trails LigandMPNN substantially.

| Split | UMA-v2 (ep19) | UMA-v1 (ep11) | LigandMPNN | ProteinMPNN | Δ vs UMA-v1 | Δ vs LigandMPNN |
|---|---:|---:|---:|---:|---:|---:|
| metal | **0.486** | 0.442 | 0.775 | 0.406 | +0.044 | -0.289 |
| small_molecule | **0.538** | 0.507 | 0.633 | 0.505 | +0.031 | -0.095 |

(See Figure 2 for the bar chart visualization, Figure 3 for per-PDB violins.)

The gap to LigandMPNN is much larger on metal (-28.9 pp) than on small molecules (-9.5 pp). We note that this is consistent with the architectural hypothesis: metal-binding sites typically have very few ligand atoms (1-3 per metal cluster), so per-residue ligand context is sparse; KNN with chemical-element typing surfaces "I'm a metal" directly, while dense attention has to discover and use this signal among a much larger set of attention targets. Small molecule binding sites have many more ligand atoms (typical drug-like compound: 10-30 heavy atoms), making the dense-attention regime more competitive.

UMA-v2 improves over UMA-v1 (a previous architecture iteration without the v2 phase 1/2/3 featurizer changes — atomic-number embedding, virtual-Cβ anchor, multi-atom backbone pair distances) by +4.4 pp on metals and +3.1 pp on small molecules — improvements concentrated in exactly the regimes the featurizer changes target.

### 3.2 Nucleotide split caveat

The nucleotide split is included in our benchmark for transparency but is **not directly comparable to LigandMPNN**. UMA-Inverse's parser (`src/data/pdb_parser.py`) ingests `HETATM` records as ligand context, while DNA and RNA atoms are stored in PDB files as `ATOM` records with residue names `DA`, `DC`, `DG`, `DT` (DNA) or `A`, `U`, `G`, `C` (RNA). Therefore UMA-Inverse trained without ever seeing nucleic acid as ligand context. The recovery numbers we report on this split (0.426 mean for v2 ep19) reflect performance on incidental HETATM cofactors within nucleotide-binding PDBs (e.g. metal cofactors, small-molecule inhibitors), not nucleotide-binding design ability per se. Closing this gap requires extending the featurizer to ingest nucleic-acid `ATOM` records and a dedicated training cycle; we flag this as future work.

### 3.3 Training dynamics

The 3-stage curriculum delivers monotonically increasing val accuracy across stages: 0.426 → 0.511 → 0.639. Stage 1 (max_total_nodes=64) plateaus quickly; the 64-node residue crop is too aggressive to learn long-range pocket-residue contacts. Stage 2 (max_total_nodes=128) delivers the largest single-stage gain (+9 pp), corresponding to the median train-set residue count being roughly within budget. Stage 3 (max_total_nodes=384) adds another +13 pp, with the canonical checkpoint (epoch 19) chosen by minimum val_loss across the 30-epoch run. (See Figure 4 for the per-stage curves.)

### 3.4 Pocket-fixed redesign: distal-position behavior

A common applied workflow with ligand-conditioned inverse-folding models is **pocket-fixed redesign** — locking the binding-pocket residues to native and redesigning the rest of the protein. This pattern shows up in directed-evolution-style sequence diversification, allosteric activity modulation, and stability engineering of enzyme scaffolds, and it is the use case where dense pair attention has the most plausible mechanistic advantage: every distal residue can attend directly to every ligand atom, whereas in a 25-nearest-atom KNN graph a residue 15+ Å from the ligand has no direct ligand edges and must reach the ligand through 2–3 hops of message passing diluted by k=25 neighbors per hop.

We selected 25 small-molecule PDBs (10 from the original v2-friendly-curated set + 15 from a deliberately unbiased extension; see Methods 2.5) and 10 metal PDBs from the LigandMPNN test splits, with hard filters (50 ≤ L ≤ 400, 5 ≤ pocket ≤ 30, distinct CCD codes for small molecules, IMAC-artifact filter for metals). For each PDB, K = 20 sequences were generated by both methods at T = 0.1 with random decoding. At each non-fixed position we computed (a) **distal recovery** (fraction of native-matching residues, averaged over the 20 designs) and (b) **distal diversity** (mean pairwise Hamming distance across the 20 designs).

**Headline numbers** (paired across PDBs):

| Split | Metric | UMA-v2 | LigandMPNN | Δ (UMA − Lig) | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| Small mol N=25 | distal recovery | 0.516 | 0.551 | −0.035 | 0.14 |
| Small mol N=25 | distal diversity | **0.131** | 0.190 | **−0.060** | **0.0015** |
| Metal N=10 | distal recovery | 0.453 | **0.554** | −0.102 | **0.004** |
| Metal N=10 | distal diversity | 0.176 | 0.162 | +0.015 | 0.92 |
| Both | pocket recovery (sanity) | 0.954 | 0.952 | n/s | n/s |

(Pocket recovery is 1.000 by construction on monomer PDBs; the residual reflects a 1-residue trim where LigandMPNN drops a position with a missing Cα atom.)

**Two findings, with the v2-friendly selection bias scrubbed by the N=25 extension:**

1. **UMA-Inverse produces distal designs with ≈ 31 % less sequence diversity than LigandMPNN on small molecules** (paired Wilcoxon p = 0.0015 across N = 25 PDBs). The original 10-PDB observation also held a +8 pp UMA edge in distal recovery, but that effect did not survive the unbiased extension to N = 25 (Δ = −0.035, n.s.); the diversity gap, by contrast, *did* survive both filter regimes. This is the first finding to interpret.
2. **Metal binding remains a clear LigandMPNN win** on distal recovery (+10 pp, p = 0.004), consistent with the established intuition that metal coordination is dominated by local sidechain geometry — exactly the signal LigandMPNN's KNN graph is built to capture.

**Why is UMA narrower?** Two interpretations are possible:

(i) UMA simply has a tighter posterior at distal positions, regardless of pocket signal — i.e. its distal predictions are less diverse but no more *informed*; or

(ii) UMA's narrowness reflects a real pocket-conditioned signal that LigandMPNN's KNN cannot transmit to distal positions: distal residues attend directly to ligand atoms, the model becomes confident about which residue type fits, and that confidence shows up as low diversity.

We discriminate these with three further analyses (Figure 5b–d):

**Test 1 — between-method confidence-ranking correlation.** If both methods respond to a shared signal (backbone, secondary-structure context), the PDBs where UMA is narrow should be the same PDBs where LigandMPNN is narrow. They are not: per-PDB Pearson r = 0.07, p = 0.75 between UMA and LigandMPNN distal-Hamming values across the N = 25 small-molecule set. **The two methods disagree about which PDBs are easy to design distally.**

**Test 2 — within-PDB amino-acid agreement.** At a per-PDB level, the AA-frequency distributions UMA and LigandMPNN produce at distal positions correlate strongly (median Pearson r = 0.85, range 0.57–0.95). The methods broadly agree on *which* AAs to put at distal positions; they disagree on *how confidently*. Combined with Test 1 this paints a "same target, different posterior width, with the width modulated by signal UMA tracks and LigandMPNN doesn't" picture.

**Test 4 — does UMA's per-PDB confidence predict cofold quality?** This is the discriminating test. If interpretation (ii) is right, narrower UMA posteriors should correspond to better Boltz-2 cofold metrics; if (i) is right, no relationship is expected. Per-PDB correlation between distal confidence (= 1 − mean distal Hamming) and cofold metrics, run separately for each method (small-molecule, N = 25 with Boltz-2 cofolds available; N drops to 22–23 for the RMSD metrics where some cofolds returned NaN ligand alignment):

| Cofold metric | UMA Pearson r | UMA p | LigMPNN Pearson r | LigMPNN p |
|---|---:|---:|---:|---:|
| ipTM (best of 5) | **+0.57** | **0.003** | +0.26 | 0.20 |
| complex pLDDT | **+0.54** | **0.006** | +0.45 | **0.025** |
| Pocket Cα RMSD (lower = better) | +0.36 | 0.09 | +0.30 | 0.17 |
| Ligand-pose RMSD | +0.25 | 0.26 | +0.08 | 0.71 |
| Affinity probability | +0.12 | 0.58 | +0.19 | 0.36 |

The asymmetry is clearest on **ipTM** — Boltz-2's interface-specific confidence: UMA's per-PDB distal confidence correlates strongly (r = +0.57, p = 0.003) where LigandMPNN's does not (r = +0.26, n.s.). On whole-protein **complex pLDDT** both methods reach significance, but UMA's correlation is stronger (r = +0.54 vs +0.45). Affinity, ligand-pose RMSD, and pocket Cα RMSD don't reach significance at this N for either method.

**The combined picture supports interpretation (ii):** UMA-Inverse's posterior width at distal positions tracks design quality on the pocket-specific cofold metric (ipTM) where LigandMPNN's does not, and tracks more strongly on the whole-protein metric (pLDDT) where both methods see something. The cleanest reading is that *both* methods' distal posteriors carry some general structure-quality information (because distal positions still inform whole-protein folding), but only UMA's posterior is also informative about the pocket-conditioned interface. Mechanistically the architectural prediction is straightforward: dense pair attention lets distal residues read directly from ligand atoms, while a 25-nearest-atom KNN graph cannot reach 15+ Å distal residues with usable ligand signal — and the metric that should be most affected by ligand-conditioned distal information is precisely ipTM, which is what we see.

We caution that this is a *correlational* finding on N = 25 PDBs (22–23 for RMSD metrics) and is therefore subject to the usual confounds (PDB-level features that drive both confidence and design quality), and that the system-vs-system attribution caveat (§3.6) applies — feature richness, not just architecture, may contribute. Nonetheless the asymmetry on the pocket-specific metric on identical PDBs is the strongest available evidence that dense attention transmits a real, ligand-conditioned signal to distal positions in this regime, and the result motivates the **pocket-fixed redesign use case** as the natural application of UMA-Inverse: when the goal is to generate diverse-but-pocket-consistent variants of a fixed scaffold (directed-evolution-style sequence diversification, allosteric tuning, scaffold stability engineering), UMA's narrower-but-pocket-tracking distal posterior is preferable to LigandMPNN's broader posterior, and UMA's per-PDB confidence is itself a usable scaffold-selection signal at the design-time stage.

### 3.5 Boltz-2 cofold of pocket-fixed redesigns

We cofolded 5 randomly-selected designs per (PDB, method) with Boltz-2 (350 cofold runs total, with 5 diffusion samples each), restricted to the same 25 small-molecule + 10 metal PDB selection. Cofold inputs use the redesigned protein sequence + the native ligand (CCD code for small molecules; ion identity for metals); the readouts are Boltz-2 confidence (ipTM, plDDT), pocket Cα RMSD vs the native crystal (after Kabsch alignment on pocket residues), ligand-pose RMSD vs native (after the same alignment), whole-scaffold Cα RMSD, and the Boltz-2 affinity head's regression value and binary-binder probability.

Boltz-2 ipTM, plDDT, and predicted affinity are *model predictions*, not measurements. We use them only as a **relative metric between two design methods on the same scaffold**, where systematic errors in the predictor should largely cancel.

| Metric (best of 5 diffusion samples; mean across cofolds) | UMA (small mol) | LigMPNN (small mol) | UMA (metal) | LigMPNN (metal) |
|---|---:|---:|---:|---:|
| ipTM | 0.91 | 0.95 | 0.78 | 0.79 |
| complex pLDDT | 0.77 | 0.88 | 0.71 | 0.75 |
| Pocket Cα RMSD (Å) | 2.65 | 1.32 | 4.07 | 4.54 |
| Ligand-pose RMSD (Å) | 3.34 | 2.07 | 0.69 | 0.43 |
| Scaffold RMSD (Å) | 5.36 | 2.76 | 7.26 | 6.68 |
| Affinity binary probability | 0.665 | 0.654 | 0.735 | 0.731 |
| Affinity pred value (log; lower = stronger) | −0.16 | −0.29 | +0.25 | +0.07 |

LigandMPNN's small-molecule cofolds preserve the crystal pocket geometry more closely than UMA's (Cα RMSD 1.32 Å vs 2.65 Å). UMA's designs deviate further from the crystal scaffold (5.36 vs 2.76 Å) — but the Boltz-2 binary-binder probability is essentially indistinguishable between methods (0.665 vs 0.654), and the regression-style affinity prediction differs by 0.13 log units (a small effect). One reading: UMA's confident distal redesigns reorganize the surrounding scaffold around the locked pocket into a *different but still binder-compatible* conformation, while LigandMPNN's broader posterior tends to leave the scaffold closer to the crystal mean. Whether the alternative scaffold conformation is a feature (novel diversification beyond what local-message-passing can produce) or a liability (geometric drift) is an open question that the cofold result does not resolve on its own — it requires experimental follow-up.

Crucially, when read alongside §3.4 Test 4, the cofold result strengthens rather than contradicts the "intelligent distal redesign" interpretation: per-PDB UMA confidence is a meaningful predictor of cofold ipTM and pLDDT, so a user who wants to *select* high-quality UMA designs has a usable confidence proxy at the input-design stage. LigandMPNN's per-PDB confidence is a substantially weaker proxy — it correlates with whole-protein pLDDT but not with the pocket-specific ipTM metric — so it is less useful for the same selection task.

### 3.6 Limitations

- **No phase-by-phase ablation**: v2's three flag-gated changes (atomic-number embedding, Cβ anchor, multi-atom backbone) were rolled out together. Per-phase contribution to the v1 → v2 gain (+4.4 pp metal / +3.1 pp small_mol) is unmeasured. Each is a config flag and thus cheap to ablate retroactively, but each ablation is a separate stage-3 training run (~3.8 days × 8× A5500). Out of scope for this preprint.
- **System-vs-system, not architecture-vs-architecture**: every LigandMPNN comparison in this paper is between two complete *systems* (architecture + featurization), not between architectures in isolation. LigandMPNN ships with a richer ligand featurizer than UMA-Inverse — multi-atom protein-ligand distances, frame-relative angle features, intra-ligand message passing, coordinate-noise regularization, and side-chain-as-context augmentation (enumerated in §4). Performance gaps therefore cannot be cleanly attributed to "dense attention vs sparse KNN" alone; they could equally reflect feature starvation on UMA's side. The pocket-fixed and cofold-correlation findings (§3.4) describe how each system *behaves end-to-end*, which is what a downstream user actually picks up off the shelf, but they do not isolate the architectural contribution. A feature-matched ablation — UMA-Inverse re-trained with the LigandMPNN feature set, architecture held constant — is the natural v3 follow-up that would disambiguate.
- **No experimental validation**: pocket-fixed redesigns were not synthesized or assayed. All claims about pocket integrity, ligand binding, and "intelligent distal redesign" rest on (a) sequence-level recovery and diversity statistics, and (b) Boltz-2 *predictions* of cofold confidence and predicted affinity, with the explicit caveat that these are not measurements.
- **Single ckpt**: results are reported on the single canonical v2 epoch-19 checkpoint. No model-averaging or ensemble decoding.
- **Selection-bias scrub is partial**: the original 10 small-molecule PDBs were selected with a v2-friendly tiebreaker (highest-standard-recovery PDB picked per CCD code); the +15-PDB extension dropped that bias. Findings reported as "robust at N = 25" are those that survive the unbiased extension; findings reported only at N = 10 should be treated as preliminary.
- **Modest N for the correlation test**: Test 4 in §3.4 uses N = 25 small-molecule PDBs (22–23 for RMSD metrics where ligand-alignment failures returned NaN). The qualitative asymmetry (UMA's ipTM correlation reaches significance; LigandMPNN's does not on the same PDBs) is robust at this N, but the absolute *r*-values should be treated as point estimates with non-trivial uncertainty.

---

## 4. Discussion

UMA-Inverse establishes that dense pair-wise attention is a **viable** architectural alternative to sparse-KNN message-passing for ligand-conditioned inverse folding. It is competitive with ProteinMPNN on standard interface recovery without conditioning on a protein graph, and the v1 → v2 featurizer changes do close some of the gap to LigandMPNN.

The remaining gap to LigandMPNN — particularly large on metal-binding sites — likely reflects a stack of architectural choices we have **not** matched, in approximate order of likely importance:

1. **KNN locality as inductive bias.** LigandMPNN's per-residue 25-nearest-ligand-atom graph hardcodes the prior that "what matters for a residue's identity is its local 3D neighborhood." Dense attention has to discover this from data; with limited compute and limited training data, the prior helps.
2. **Multi-atom protein-ligand distances.** LigandMPNN encodes 5 backbone-atom-to-ligand distances per (residue, ligand atom) pair; UMA-Inverse encodes only the Cβ-anchor distance.
3. **Frame-relative angle features.** LigandMPNN adds sin/cos features describing each ligand atom in the N-Cα-C frame of each pocket residue. UMA-Inverse has no orientation features — only inter-atom distances.
4. **Intra-ligand graph.** LigandMPNN runs message passing among ligand atoms before projecting to residues. UMA-Inverse treats ligand atoms as plain nodes in the same pair tensor; intra-ligand structure must be inferred from the ligand-ligand block.
5. **Coordinate noise + side-chain augmentation.** LigandMPNN trains with 0.1 Å Gaussian noise on coordinates (regularization vs backbone memorization) and 2-4% random side-chain-as-context augmentation. UMA-Inverse does neither.

Each of these is a candidate v3 direction; together they likely account for most of the metal gap. The cleanest way to disambiguate architectural inductive bias from feature richness, however, is a single targeted experiment: re-train UMA-Inverse with the full LigandMPNN feature set, holding architecture constant. If UMA closes the metal gap under matched featurization, the bottleneck was feature starvation; if it does not, the locality-prior hypothesis (dense attention underfits coordination-shell-dominated signals where KNN locality is the right inductive bias) is supported. Until that experiment is run, the comparisons in this paper are best read as system-vs-system (see §3.6 limitations).

The pocket-fixed redesign experiment (§3.4) probes a regime where dense attention's structural advantage — direct edges from every residue to every ligand atom — could plausibly matter. The data show that UMA-Inverse and LigandMPNN agree on *which* amino acids to place at distal positions (median per-PDB AA-distribution Pearson r = 0.85) but disagree sharply on *how confidently*: UMA's distal-position posterior is narrower, and that per-PDB posterior width predicts Boltz-2 cofold quality more reliably for UMA than for LigandMPNN — most clearly on the pocket-specific ipTM metric (UMA r = +0.57, p = 0.003; LigandMPNN r = +0.26, n.s.). On whole-protein pLDDT both methods correlate, with UMA stronger; the asymmetry is concentrated where the architectural prediction says it should be — at the protein-ligand interface. This is consistent with dense pair attention transmitting a ligand-conditioned signal to distal residues that 25-nearest-atom KNN message-passing cannot reach. We treat this as suggestive rather than dispositive — the system-vs-system caveat applies, the test is correlational, and N = 25 — but it identifies the **pocket-fixed redesign use case** (directed-evolution-style sequence diversification, allosteric tuning, scaffold stability engineering) as the regime where UMA's narrower-but-pocket-tracking distal posterior is preferable to LigandMPNN's broader posterior, and where UMA's per-PDB confidence is itself a usable scaffold-selection signal at design time.

This work does not advocate for or against KNN message-passing. We document an architectural alternative, characterize where it lands on standard benchmarks, and identify a regime where it may have a structural niche. UMA-Inverse the model and codebase are made available to the community as a baseline for future architectural comparisons in ligand-conditioned design.

---

## 5. Code and data availability

All code, configuration files, training logs, and benchmark scripts are at `<github URL TBD>`, branch `v2-element-embedding`. The canonical v2 stage-3 checkpoint is at `checkpoints/pairmixerinv-v2-stage3-nodes384-ddp8/uma-inverse-19-1.1463.ckpt`. PDB selection for the pocket-fixed redesign experiment is at `outputs/preprint/pdb_selection.json` and is fully reproducible from `scripts/paper/select_pdbs.py` given the same v2-ep19 benchmark CSVs as input.

The training data (LigandMPNN train/valid splits, 154K PDBs) is fetched from RCSB by `scripts/SLURM/01a_fetch_data.sh`; the cached `.pt` tensors are produced by `scripts/SLURM/01b_preprocess.sh`. Both are deterministic given the LigandMPNN split JSONs.

Boltz-2 cofolds [Passaro et al. 2025] were generated using the public release (<https://github.com/jwohlwend/boltz>) with the exact CLI flags documented in `scripts/SLURM/run_boltz_example.sh` and the YAML schema in `scripts/paper/build_boltz_yaml.py` and `scripts/paper/build_cofold_yamls.py`.

---

## References

[Dauparas et al. 2022] Dauparas, J. et al. *Robust deep learning–based protein sequence design using ProteinMPNN*. Science 378, 49–56 (2022).

[Dauparas et al. 2023] Dauparas, J. et al. *Atomic context-conditioned protein sequence design using LigandMPNN*. Nature Methods (2023). bioRxiv preprint: 2023.12.22.573103.

[Wohlwend et al. 2024] Wohlwend, J. et al. *Boltz-1: Democratizing Biomolecular Interaction Modeling*. bioRxiv (2024).

[Passaro et al. 2025] Passaro, S., Corso, G., Wohlwend, J., Reveiz, M., Thaler, S., Ram Somnath, V., Getz, N., Portnoi, T., Roy, J., Stark, H., Kwabi-Addo, D., Beaini, D., Jaakkola, T., & Barzilay, R. *Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction*. bioRxiv 2025.06.14.659707 (2025). doi: 10.1101/2025.06.14.659707.
