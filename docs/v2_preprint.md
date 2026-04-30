# Dense Pair-Wise Attention for Ligand-Conditioned Protein Sequence Design

**UMA-Inverse: a single-encoder alternative to KNN message-passing for ligand-conditioned inverse folding.**

W. Sobolewski¹

¹ Yeh Lab, University of California, Santa Cruz.
Correspondence: `wsobolew@ucsc.edu`. Code and weights: <https://github.com/...> (TBD before submission).

---

## Abstract

Existing ligand-conditioned protein sequence design methods, notably LigandMPNN, are built on top of K-nearest-neighbor message-passing graphs that limit ligand information to each residue's local protein neighborhood. We explore an alternative architecture, **UMA-Inverse**, which uses a single dense pair-wise attention encoder over the concatenated residue + ligand atom set with no KNN sparsification. UMA-Inverse outperforms ProteinMPNN on the LigandMPNN-paper interface-recovery benchmark by **+8.0 percentage points on metal-binding sites (0.486 vs 0.406)** and **+3.3 percentage points on small molecules (0.538 vs 0.505)**, but trails LigandMPNN itself, with the gap concentrated on metal sites (-28.9 pp). To characterize where dense pair-wise attention may offer a structural advantage absent from local KNN methods, we test the methods at **pocket-fixed redesign**: holding the binding-pocket residues to their native identities and asking the model to redesign the rest of the protein. Using Boltz-2 cofolding to evaluate redesigned sequences against their cognate ligands, we find <RESULT TBD: distal sequence diversity, predicted-affinity comparison, pocket-pose preservation>. We argue that dense pair-wise attention is a viable architectural alternative to KNN message-passing for ligand-conditioned design, particularly in regimes where global context propagation is needed.

---

## 1. Introduction

Designing protein sequences that fold to a specified backbone is the inverse-folding problem; conditioning on additional non-protein context (small molecules, nucleotides, metal ions) extends the problem to ligand-binding design. The de facto standard, **ProteinMPNN** [Dauparas et al. 2022], handles backbone-only design via a sparse-graph message-passing neural network with K-nearest-neighbor edges between residues. **LigandMPNN** [Dauparas et al. 2023] extends ProteinMPNN with two parallel KNN graphs over ligand atoms (a ligand-only graph and a protein-ligand bipartite graph), achieving strong interface-recovery numbers on small-molecule, nucleotide, and metal-binding test splits.

LigandMPNN's architecture inherits ProteinMPNN's locality assumption: each residue sees only its K nearest ligand atoms (default K=25), with no direct edge to ligand atoms beyond that radius. In practice this is fine for residues directly contacting a ligand, but it creates an information bottleneck for residues farther away — distal protein-side scaffolding residues that may nonetheless influence pocket geometry only see ligand context through 2-3 hops of message passing, mixed at each hop with K=25 other contributions. We hypothesize that this locality may matter for design tasks where global context propagation is important: e.g., redesigning a distal residue *to support* a fixed binding pocket.

In this work we build and evaluate **UMA-Inverse**, an alternative architecture for the same problem. Rather than three sparse KNN graphs, UMA-Inverse uses a single dense pair-wise attention encoder over the union of residue and ligand atom nodes. Every residue has direct edges to every ligand atom and every other residue; no graph sparsification. We benchmark UMA-Inverse on LigandMPNN's protocol (interface sequence recovery, 10 designs/PDB, T=0.1, random decoding order) and find that it lands between LigandMPNN and ProteinMPNN on standard recovery metrics. We then characterize a regime where the architectural difference *should* matter — pocket-fixed redesign — and find <STATEMENT OF RESULT TBD>.

The contribution is threefold:

1. A single-encoder, dense-pair-wise-attention baseline for ligand-conditioned inverse folding.
2. Like-for-like benchmark numbers under the LigandMPNN protocol, including all three test splits.
3. Empirical comparison of UMA vs LigandMPNN on pocket-fixed redesign, with structural analysis via Boltz-2 cofolding.

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

Implementation at `scripts/preprint/select_pdbs.py`. Selection persisted at `outputs/preprint/pdb_selection.json` with per-PDB pocket residue IDs, ligand identity (CCD code or ion symbol), and selection_reason for provenance.

For each selected PDB, both UMA-v2 and LigandMPNN are tasked with **redesigning the protein with the pocket residues forced to their native identity**. K=20 sequences are generated per (PDB, method) at T=0.1, random decoding order. Implementation at `scripts/preprint/run_pocket_fixed_designs.py` (UMA, via the existing `DesignConstraints.fix` path) and `scripts/preprint/build_ligandmpnn_inputs.py` + `scripts/SLURM/preprint_ligandmpnn_pocket_fixed.sh` (LigandMPNN, via `LigandMPNN/run.py --fixed_residues_multi`).

Per (PDB, method, sample) metrics:
- **Distal recovery**: fraction of non-fixed positions where predicted == native.
- **Pocket recovery**: fraction of fixed positions where predicted == native (sanity check, expected 1.0).

Per (PDB, method) aggregates:
- Mean / median / stdev of distal recovery.
- Mean pairwise Hamming distance at distal positions across the K samples (sequence diversity).
- Per-AA frequency at distal positions.

Statistics: paired Wilcoxon signed-rank test (UMA vs LigandMPNN) on per-PDB summary metrics, with bootstrap 95% CI on the mean paired difference.

### 2.5 Boltz-2 cofold of redesigned sequences

Five sequences per (PDB, method) are randomly subsampled (= 200 cofold inputs total). Each (sequence, ligand) pair is cofolded by **Boltz-2** [Wohlwend et al. 2024] using protein sequence + ligand CCD code as input, with affinity prediction enabled (`--sampling_steps_affinity 200 --diffusion_samples_affinity 5`). Boltz-2 outputs a predicted complex structure (PDB) plus a confidence JSON with ipTM, plDDT, and affinity scores.

Per (PDB, method, sample, diffusion_sample) metrics:
- **Boltz-2 confidence**: ipTM (interface predicted TM-score), plDDT (whole-protein), plDDT restricted to original-pocket residues, plDDT restricted to interface residues in the cofold.
- **Boltz-2 predicted affinity**: binary classification + pKi-style score from the affinity head.
- **Pocket geometry preservation**: pocket Cα RMSD between Boltz-2 cofold and native crystal, restricted to the fixed pocket residues.
- **Ligand pose preservation**: ligand heavy-atom RMSD vs the native ligand pose after aligning on pocket Cα.
- **Whole-protein scaffold RMSD**: Cα RMSD vs native crystal.

**Important caveat: Boltz-2 ipTM, plDDT, and affinity are model predictions, not measurements.** They are useful for relative comparisons between two design methods evaluating the same scaffold (where systematic prediction errors should largely cancel) but are not interpretable as absolute binding affinity or design quality. We do not claim binding affinities or catalytic efficiencies on the basis of these metrics; we use them as a proxy for "did the redesign produce a coherent ligand-pocket complex."

Implementation at `scripts/preprint/cofold.py` (TBD), modeled on the existing `scripts/SLURM/run_boltz_example.sh` invocation pattern with the same Boltz-2 flags.

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

### 3.4 Pocket-fixed redesign

[**TO BE FILLED IN once Phase A and B run.** Expected structure:]

- Per-PDB distal recovery distributions for UMA-v2 vs LigandMPNN. Headline: do the means differ significantly? Direction?
- Distal sequence diversity (mean pairwise Hamming distance at distal positions). Headline claim — *if* UMA shows higher distal diversity *and* comparable distal recovery, this is the architectural advantage we hypothesized.
- Boltz-2 cofold metrics: ipTM, pocket Cα RMSD, ligand pose RMSD, predicted affinity. Direction of paired comparison UMA vs LigandMPNN per metric.
- Failure analysis: any (PDB, method) pair with mean ipTM < 0.5 or ligand RMSD > 5 Å.

(Figures 5 and 6 — the contribution figures.)

### 3.5 Limitations

- **No phase-by-phase ablation**: v2's three flag-gated changes (atomic-number embedding, Cβ anchor, multi-atom backbone) were rolled out together. Per-phase contribution to the v1 → v2 gain (+4.4 pp metal / +3.1 pp small_mol) is unmeasured. Each is a config flag and thus cheap to ablate retroactively, but each ablation is a separate stage-3 training run (~3.8 days × 8× A5500). Out of scope for this preprint.
- **No experimental validation**: pocket-fixed redesigns were not synthesized or assayed. All claims about pocket integrity / ligand binding are based on Boltz-2 *predictions*, with the explicit caveat that these are not measurements.
- **Single ckpt**: results are reported on the single canonical v2 epoch-19 checkpoint. We did not perform model-averaging or ensemble decoding.
- **Selection bias in pocket-fixed PDB choice**: the 20-PDB selection was filtered for v2-friendly properties (high standard-recovery PDBs preferred) — the pocket-fixed numbers may overstate UMA's typical performance on a uniformly-sampled set.

---

## 4. Discussion

UMA-Inverse establishes that dense pair-wise attention is a **viable** architectural alternative to sparse-KNN message-passing for ligand-conditioned inverse folding. It is competitive with ProteinMPNN on standard interface recovery without conditioning on a protein graph, and the v1 → v2 featurizer changes do close some of the gap to LigandMPNN.

The remaining gap to LigandMPNN — particularly large on metal-binding sites — likely reflects a stack of architectural choices we have **not** matched, in approximate order of likely importance:

1. **KNN locality as inductive bias.** LigandMPNN's per-residue 25-nearest-ligand-atom graph hardcodes the prior that "what matters for a residue's identity is its local 3D neighborhood." Dense attention has to discover this from data; with limited compute and limited training data, the prior helps.
2. **Multi-atom protein-ligand distances.** LigandMPNN encodes 5 backbone-atom-to-ligand distances per (residue, ligand atom) pair; UMA-Inverse encodes only the Cβ-anchor distance.
3. **Frame-relative angle features.** LigandMPNN adds sin/cos features describing each ligand atom in the N-Cα-C frame of each pocket residue. UMA-Inverse has no orientation features — only inter-atom distances.
4. **Intra-ligand graph.** LigandMPNN runs message passing among ligand atoms before projecting to residues. UMA-Inverse treats ligand atoms as plain nodes in the same pair tensor; intra-ligand structure must be inferred from the ligand-ligand block.
5. **Coordinate noise + side-chain augmentation.** LigandMPNN trains with 0.1 Å Gaussian noise on coordinates (regularization vs backbone memorization) and 2-4% random side-chain-as-context augmentation. UMA-Inverse does neither.

Each of these is a candidate v3 direction; together they likely account for most of the metal gap.

The pocket-fixed redesign experiment (§3.4) probes a regime where dense attention's structural advantage — direct edges from every residue to every ligand atom — could plausibly matter. <STATEMENT OF EMPIRICAL FINDING TBD: whether the architectural difference manifests as measurable distal-design diversity / pocket-conditioning behavior>.

This work does not advocate for or against KNN message-passing. We document an architectural alternative, characterize where it lands on standard benchmarks, and identify a regime where it may have a structural niche. UMA-Inverse the model and codebase are made available to the community as a baseline for future architectural comparisons in ligand-conditioned design.

---

## 5. Code and data availability

All code, configuration files, training logs, and benchmark scripts are at `<github URL TBD>`, branch `v2-element-embedding`. The canonical v2 stage-3 checkpoint is at `checkpoints/pairmixerinv-v2-stage3-nodes384-ddp8/uma-inverse-19-1.1463.ckpt`. PDB selection for the pocket-fixed redesign experiment is at `outputs/preprint/pdb_selection.json` and is fully reproducible from `scripts/preprint/select_pdbs.py` given the same v2-ep19 benchmark CSVs as input.

The training data (LigandMPNN train/valid splits, 154K PDBs) is fetched from RCSB by `scripts/SLURM/01a_fetch_data.sh`; the cached `.pt` tensors are produced by `scripts/SLURM/01b_preprocess.sh`. Both are deterministic given the LigandMPNN split JSONs.

Boltz-2 cofolds, used for pocket-fixed evaluation, were generated using the Boltz-2 install at `<TBD>` with the flags documented in `scripts/SLURM/run_boltz_example.sh`.

---

## References

[Dauparas et al. 2022] Dauparas, J. et al. *Robust deep learning–based protein sequence design using ProteinMPNN*. Science 378, 49–56 (2022).

[Dauparas et al. 2023] Dauparas, J. et al. *Atomic context-conditioned protein sequence design using LigandMPNN*. Nature Methods (2023). bioRxiv preprint: 2023.12.22.573103.

[Wohlwend et al. 2024] Wohlwend, J. et al. *Boltz-1: Democratizing Biomolecular Interaction Modeling*. bioRxiv (2024).

(Boltz-2 reference TBD; the model's affinity head publication forthcoming.)
