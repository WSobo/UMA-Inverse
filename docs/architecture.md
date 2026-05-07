# UMA-Inverse: Architecture & Science

> **Context:** UMA-Inverse is a ligand-conditioned protein inverse-folding model — given a fixed protein-ligand backbone, it predicts what amino acid sequence best complements that structure. It upgrades LigandMPNN's graph-message-passing backbone with PairMixer, a dense pair-representation encoder.

---

## 1. The Problem: Ligand-Conditioned Inverse Folding

Standard inverse folding asks: *given a protein backbone, what sequence folds into it?* The ligand-conditioned variant adds the crucial constraint: *given a protein-ligand complex backbone, what sequence folds into it **and** binds the ligand?*

```
  Input:                              Output:
  ┌────────────────────────────┐      ┌────────────────────┐
  │  Cα coords  [L, 3]         │      │  AA logits [L, 21] │
  │  Backbone φ/ψ/ω  [L, 6]   │ ───▶ │  21 = 20 AA + X    │
  │  Ligand heavy atoms [M, 3] │      └────────────────────┘
  │  Ligand element types [M]  │
  └────────────────────────────┘
```

This is relevant for **drug binding site design**: engineer a protein whose binding pocket wraps snugly around a given small molecule.

### Comparison to LigandMPNN

LigandMPNN uses a **sparse graph** (k-NN edges) with message-passing GNNs. Each residue aggregates information from its k nearest neighbours over several rounds. Ligand atoms are treated as extra graph nodes.

**The tradeoff:** sparse graphs are fast but have limited "long-range path length" — distant residues only interact through chains of local edges.

---

## 2. The PairMixer Idea (ICLR 2026 Source Paper)

AlphaFold3 uses a **Pairformer** backbone: a dense N×N pair tensor z where z_ij encodes the relationship between residue i and residue j. Updates use:
- Triangle Self-Attention (O(N³) memory — the bottleneck)
- Triangle Multiplication (O(N²) via matmul)
- Transition MLP

The Genesis/Pearl paper showed you can **drop triangle attention entirely** and keep only triangle multiplication + MLP. This gives:
- 4× faster inference on long sequences
- 34% less training cost
- Essentially no accuracy drop on folding/docking benchmarks

**Why does it still work?** Triangle multiplication already encodes the key geometric inductive bias. Triangle attention just adds learnable routing that the model can compensate for with more parameters.

---

## 3. UMA-Inverse Architecture (Full Data Flow)

```
 PDB file
    │
    ▼ pdb_parser.py (BioPython)
 X[L,4,3]  S[L]  Y[M,3]  Y_t[M]
    │               │
    ▼ ligandmpnn_bridge.py
 residue_coords [L,3]   ligand_coords [M,3]
 residue_features[L,6]  ligand_features[M,6]
    │  (sin/cos φ,ψ,ω)    │  (element one-hot C/N/O/S/P/X)
    │                      │
    └──────────────────────┘
                │
    ┌───────────▼──────────────┐
    │    Node Initialization   │
    │  res: Linear(6 → 128)   │
    │  lig: Linear(6 → 128)   │
    │  cat + LayerNorm         │
    │  node_repr [B, L+M, 128] │
    └───────────┬──────────────┘
                │
    ┌───────────▼──────────────────────────────────┐
    │         Pair Tensor Initialization           │
    │                                              │
    │  z_ij = node_i + node_j + RBF(dist_ij)      │
    │                                              │
    │  ├─ node_i: Linear(128→128)[B,L+M,1,128]    │
    │  ├─ node_j: Linear(128→128)[B,1,L+M,128]    │
    │  └─ RBF: 32 Gaussians → Linear(32→128)      │
    │                                              │
    │  z [B, L+M, L+M, 128]   (masked)            │
    └───────────┬──────────────────────────────────┘
                │
    ┌───────────▼──────────────┐
    │   PairMixer Encoder      │
    │   (6 blocks)             │
    │                          │
    │  ┌────────────────────┐  │
    │  │  PairMixerBlock    │  │
    │  │  ─────────────     │  │
    │  │  z += TriMulIn(z)  │  │
    │  │  z += TriMulOut(z) │  │
    │  │  z += Transition(z)│  │
    │  └────────────────────┘  │
    │         × 6              │
    │                          │
    │  z [B, L+M, L+M, 128]    │
    └─────┬──────────┬─────────┘
          │          │
          ▼          ▼
   LigandContext   AR Context
   (mean pool)     (causal attn)
      [B,L,128]    [B,L,128]
          │          │
          └────┬─────┘
               ▼
    ┌──────────────────────────┐
    │  Decoder                 │
    │  cat([node + ar, lig_ctx])│
    │  LayerNorm(256)          │
    │  Linear(256 → 128)       │
    │  GELU                    │
    │  Linear(128 → 21)        │
    └──────────┬───────────────┘
               ▼
          logits [B, L, 21]
```

---

## 4. Triangle Multiplication: The Core Geometric Primitive

The key intuition is that **three-body relationships** (i, j, k triplets) can be captured by factoring the pair update through an intermediary k.

### Outgoing (shared emanating edges)

```
      k
     / \
    i   j     z_ij ← Σ_k  gate(z_ik) · gate(z_jk)

  "i and j are related because they both connect to k"
```

Implementation:
```python
# left[b,i,k,h] and right[b,j,k,h] from z_norm
left_p  = left.permute(0,3,1,2)   # [B, H, i, k]
right_p = right.permute(0,3,2,1)  # [B, H, k, j]
z_out = left_p @ right_p          # [B, H, i, j]  ✓ math checks out
```

### Incoming (shared incident edges)

```
     i   j
      \ /
       k     z_ij ← Σ_k  gate(z_ki) · gate(z_kj)

  "i and j are related because they both receive from k"
```

Implementation:
```python
left_p  = left.permute(0,3,2,1)   # [B, H, i, k]  (swapped k↔i)
right_p = right.permute(0,3,1,2)  # [B, H, k, j]
z_out = left_p @ right_p          # [B, H, i, j]  ✓
```

Both directions together give the pair tensor a chance to reason about all geometric triplets in the structure — critical for capturing the protein-ligand interaction geometry.

---

## 5. Autoregressive Decoder

Inverse folding is **autoregressive** during training: the identity of previously-decoded residues conditions predictions for later ones. The decoding order is determined by a seeded random shuffle (non-designed residues first, then designed ones in random order).

```
Step 1: Sample decoding_order π for this (epoch, batch_idx)
        non-designed residues get priority (decode first)

Step 2: For each residue i in π:
        attention_weight[i,j] = softmax(z_ij / √d)[j < i in π]
                                          ↑
                                 causal: only see already-decoded j

Step 3: ar_context[i] = Σ_{j decoded before i} weight[i,j] · token_emb[j]

Step 4: logits[i] = Decoder(node_repr[i] + ar_context[i], ligand_context[i])
```

The seeding strategy `seed = epoch * 1_000_000 + batch_idx` gives:
- **Reproducibility** — same data → same decoding order in every run
- **Diversity** — different batches/epochs see different orders, preventing the model from overfitting to a fixed order
- **Correct conditioning** — fixed residues (design_mask=False) are always decoded "first", so they provide sequence context to designed positions

---

## 6. Input Feature Details

### Residue Features (6-dim)

Backbone dihedral angles encode local secondary structure:

| Index | Feature | Meaning |
|-------|---------|---------|
| 0 | sin φ | N-Cα-C-N torsion |
| 1 | sin ψ | Cα-C-N-Cα torsion |
| 2 | sin ω | C-N-Cα-C peptide bond planarity |
| 3 | cos φ | (same, cosine component) |
| 4 | cos ψ | |
| 5 | cos ω | |

### Ligand Features (6-dim)

Heavy atom element one-hot:

```
[ C, N, O, S, P, other ]
  6  7  8  16 15
```

### RBF Distance Embedding

32 Gaussian basis functions centered at `linspace(0, 24 Å, 32)`:

```
φ_k(d) = exp(-γ · (d - μ_k)²)      γ = (24/31)^{-2}

d (Å):  0    4    8   12   16   20   24
        ■■   ■■   ■■   ■■   ■■   ■■   ■■
```

---

## 7. Potential Issues and Peculiarities

### 🔴 Critical: Node Representations Are Never Updated by the Encoder

This is the most significant architectural concern.

The encoder **only updates the pair tensor z**. The `node_repr` seen by the decoder is the same initial linear embedding — it never receives feedback from the 6 PairMixer blocks. All structural information from the encoder reaches the decoder only through the mean-pooled `ligand_context`:

```python
# Encoder updates z but NOT node_repr
z = self.encoder(z, pair_mask)

# The only way encoder knowledge reaches decoder:
ligand_context = self._ligand_aware_context(z, ...)  # mean pool over dim=2
# → [B, L, 128] — all 128-dim pair information collapsed into one vector per residue

# node_repr is the RAW INPUT EMBEDDING, pre-encoder
decoder_input = cat([node_repr[:, :L, :] + ar_context, ligand_context], dim=-1)
```

In AlphaFold3, the pair representation continuously feeds back into sequence-level representations via attention with pair bias. Here, 6 blocks of pair refinement are compressed into a single mean pool. **This likely degrades model capacity.**

**Suggested fix:** After the encoder, extract per-residue node representations from z by summing over the pair dimension (e.g., `z_mean = z[:, :L, :, :].mean(dim=2)`) and adding it back to `node_repr` before the decoder. Or add a proper "pair-to-node" update module.

---

### 🟠 Important: AR Context Ignores Residue–Ligand Pairs

The autoregressive context only looks at the residue-residue subblock of z:

```python
rr = z[:, :residue_count, :residue_count, :]  # [B, L, L, 128]
score = self.ar_pair_to_scalar(rr).squeeze(-1)  # [B, L, L]
```

The residue-ligand block `z[:, :L, L:, :]` is ignored in the AR attention. This means the autoregressive context has no direct access to *how much the ligand influences the decoding order/dependencies* — it's as if ligand atoms don't exist during the AR step.

The ligand does influence things indirectly because the encoder updated all of z (including rr) based on ligand-pair interactions. But using ligand-residue pair information directly in the AR attention weights could improve binding-site residue predictions.

---

### 🟡 Moderate: "Ligand-Aware" Context is Actually Residue-Dominated

```python
# z_res: pair features for all residue rows, columns = ALL nodes (protein + ligand)
z_res = z[:, :residue_count, :, :]   # [B, L, L+M, 128]
weights = pair_mask[:, :residue_count, :]  # [B, L, L+M]
ligand_context = (z_res * w).sum(dim=2) / w.sum(dim=2)
```

This averages across **all** nodes (residues AND ligand). For a typical structure with L=300 residues and M=25 ligand atoms, the ligand contributes only ~8% of the average. The function is called "ligand-aware" but is dominated by residue-residue pair information.

**Suggested fix:** Compute separate mean pools for the residue-residue and residue-ligand blocks, concatenate them, and project to pair_dim. This would give residue-specific ligand context while retaining protein structural context.

---

### 🟡 Moderate: No Bottleneck in Triangle Multiplication

```yaml
# configs/config.yaml
pair_dim: 128
pair_hidden_dim: 128   # ← same as pair_dim
```

In AlphaFold2/3, `hidden_dim` is typically ≤ `pair_dim` to create a low-rank bottleneck (the geometric information is compressed through a lower-dimensional intermediate). With `hidden_dim = pair_dim = 128`, the triangle multiplication performs a full-rank update with 2× the parameters. This isn't wrong, but:
- Doubles the parameter count in TriMul layers vs. a 64-dim bottleneck
- Loses the low-rank inductive bias (that pair updates should be "simple" geometric transformations)
- More memory-hungry for longer sequences

---

### 🟡 Moderate: Residue Features are Sparse

Only 6 backbone dihedral features. LigandMPNN uses richer geometric features:
- Virtual Cβ positions
- Local frame orientations  
- Distance to chain termini
- Secondary structure classifications

The PairMixer encoder can partially compensate because the RBF-encoded pairwise distances carry structural context. But the initial node features provide the seeds for the encoder — sparse seeds may limit expressivity.

---

### 🟢 Minor: Thermal Noise is Off by Default

`thermal_noise_std: 0.0` — adding small Gaussian noise to coordinates during training is a useful regularizer that simulates crystallographic B-factors. The infrastructure is there; set to ~0.05–0.2 Å to encourage robustness to coordinate uncertainty.

---

### 🟢 Minor: Decoding Order Not Used at Inference

During inference (`sequence=None` in `_autoregressive_context`), the AR context is zeroed out entirely:

```python
if sequence is None:
    return torch.zeros((B, L, node_dim), ...)
```

This means at inference time the model makes **all predictions simultaneously**, with no autoregressive conditioning between residues — it's effectively a one-shot predictor. This inconsistency with training (where AR conditioning is used) is standard in practice (same as ProteinMPNN/LigandMPNN), but it means the model must learn to be good both with and without AR context.

---

## 8. Suggested Improvements

### ~~Priority 1: Add a Pair-to-Node Readout~~ ✅ Implemented

`self.node_readout = Linear(pair_dim, node_dim)` added to `__init__`; `node_repr_res = node_repr[:, :L, :] + self.node_readout(z_pooled)` wired in `forward()` before the decoder.

### ~~Priority 2: Separate Ligand vs. Protein Context~~ ✅ Implemented

`self.ctx_proj = Linear(2 * pair_dim, pair_dim)` replaces the old single mean-pool. `_ligand_aware_context` now computes separate `rr_ctx` and `rl_ctx` and projects them through `ctx_proj`.

### Priority 3: Wider RBF + Larger max_distance for Drug-like Ligands

24 Å is borderline for large drug-like molecules. Allosteric sites can be >20 Å from the nearest ligand atom. Consider `max_distance: 32.0` and `num_rbf: 48`.

### Priority 4: pair_hidden_dim = pair_dim / 2

Set `pair_hidden_dim: 64` for a proper low-rank bottleneck. This halves TriMul parameters and may improve generalization (less overfitting to training structures).

---

## 9. Is It Ready to Test?

**Yes, with caveats.**

The core architecture is mathematically sound:
- Triangle multiplication implementations are correct (both outgoing and incoming permute ops verified)
- AR masking logic handles edge cases (first residue, all-fixed)
- Padded batches are properly masked throughout
- Gradient flow is intact through the full pipeline

The 20 integration tests cover all shape contracts and gradient flows on CPU. The natural next validation is the pilot overfit test (`make pilot`) — if the model can memorize a single structure to near-zero loss, the forward/backward pass is verified end-to-end.

**The main scientific risk** is not a bug but a capacity issue: the compressed ligand context (mean pool of all pair rows) may not provide enough signal for the decoder to distinguish binding-site residues from surface-exposed ones. This would manifest as val/loss plateauing early or the model failing to improve over LigandMPNN's baseline ~1.38 cross-entropy (native sequence perplexity ≈4).

**Recommended experiment sequence:**
1. `make pilot` — verify convergence on 1 structure (loss → ~0)
2. Stage 1 training (`max_total_nodes=64`) — verify val loss < ~2.5 after 1 epoch
3. If yes: proceed to full curriculum
4. If no: implement Priority 1 (pair-to-node readout) before proceeding

---

## 10. Parameter Count Estimate

| Component | Params |
|-----------|--------|
| residue_in Linear(6→128) | 768 + 128 |
| ligand_in Linear(6→128) | 768 + 128 |
| node_norm LayerNorm(128) | 256 |
| pair_i, pair_j Linear(128→128) | 2 × 16,384 |
| rbf_proj Linear(32→128) | 4,096 |
| 6 × PairMixerBlock | ~6 × ~530k ≈ 3.2M |
| ar_pair_to_scalar Linear(128→1) | 128 |
| token_embedding Embedding(21,128) | 2,688 |
| decoder Sequential | ~66k |
| **Total** | **~3.3M params** |

This is comparable to LigandMPNN (~2.5M) — small enough for fast experimentation, large enough to learn complex sequence-structure relationships.
