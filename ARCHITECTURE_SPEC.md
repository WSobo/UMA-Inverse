# UMA-Inverse Architecture Spec (v1)

This document is the implementation contract for UMA-Inverse: a LigandMPNN-comparable inverse folding model using a dense invariant PairMixer encoder with a single-GPU training path.

## 1) Scope and constraints

- Objective: predict amino-acid identity per residue for a fixed protein-ligand structure.
- Inputs: protein backbone coordinates + ligand atom coordinates/types.
- Outputs: per-residue logits over 21 tokens (20 AA + X).
- Data compatibility: use LigandMPNN parser and split JSON files.
- Hardware target: one 24 GB class GPU.
- Non-goals for v1:
  - no full SE(3) kernel stack
  - no sidechain coordinate head
  - no diffusion head

## 2) Repository layout

Follow UMA-Fold style:

- `configs/`: config surface for model, data, trainer.
- `scripts/`: train, inference, pilot-run, SLURM wrappers.
- `src/data/`: LigandMPNN bridge and datamodule.
- `src/models/`: PairMixer blocks and UMA-Inverse model.
- `src/training/`: LightningModule.
- `tests/`: smoke tests for shape and collation.

## 3) Data pipeline (LigandMPNN bridge)

### 3.1 Parsing

Use `parse_PDB` and `featurize` from `../LigandMPNN/data_utils.py`.

Required behavior:

- Keep identical sequence tokenization to LigandMPNN (`S` vocabulary).
- Keep residue-level valid mask (`mask`) and chain design mask (`chain_mask`).
- Read ligand atom positions and atomic numbers from `Y`, `Y_t`, `Y_m`.

### 3.2 Node features

- Protein nodes:
  - CA coordinates from `X[:, 1, :]`
  - 6D dihedral feature: `sin/cos(phi, psi, omega)` from backbone atoms N, CA, C
- Ligand nodes:
  - 6D one-hot element bins: C, N, O, S, P, other

### 3.3 Cropping policy for single GPU

- Keep nearest ligand atoms up to `ligand_context_atoms` (default 25).
- Keep total nodes below `max_total_nodes` (default 384):
  - if protein too long, choose residues nearest ligand centroid
  - if no ligand, keep leading residues

### 3.4 Batch collation

Pad variable-length tensors:

- `residue_coords`: `[B, L_max, 3]`
- `residue_features`: `[B, L_max, 6]`
- `residue_mask`: `[B, L_max]`
- `sequence`: `[B, L_max]`
- `design_mask`: `[B, L_max]`
- `ligand_coords`: `[B, M_max, 3]`
- `ligand_features`: `[B, M_max, 6]`
- `ligand_mask`: `[B, M_max]`

## 4) Model architecture

### 4.1 Initialization

Project residue and ligand node features into shared node space:

- `h_res = Linear(residue_features)`
- `h_lig = Linear(ligand_features)`
- `h = concat(h_res, h_lig)`

Build pair mask from padded node mask.

### 4.2 Invariant dense pair tensor

Construct dense pair representation `Z`:

- Pairwise distances `D_ij = ||x_i - x_j||_2`
- Radial basis expansion `rbf(D_ij)`
- Pair init:
  - `Z_ij = W_i(h_i) + W_j(h_j) + W_rbf(rbf(D_ij))`

Optional thermal augmentation:

- During training only, add small Gaussian noise to distances before RBF.

### 4.3 PairMixer block

Each block applies:

1. Triangle multiplication outgoing
2. Triangle multiplication incoming
3. Transition MLP

Residual update style:

- `Z = Z + TriOut(Z)`
- `Z = Z + TriIn(Z)`
- `Z = Z + Transition(Z)`

Implementation notes:

- Use `torch.einsum` kernels for triangle multiplication.
- Apply pair mask after each sub-update.
- Use gradient checkpointing across blocks when training.

### 4.4 Decoder

Residue logits use two contexts:

- ligand-aware context: aggregate residue-to-all-node pair embeddings
- autoregressive teacher-forced context: causal weighted sum over previously decoded residue token embeddings

Final input per residue:

- `concat(node_repr_i + ar_context_i, ligand_context_i)`

Then MLP to logits over 21 tokens.

## 5) Loss and optimization

### 5.1 Loss

Cross entropy on designed residues only:

- valid positions = `residue_mask & design_mask`
- ignore padded or fixed positions with target `-100`

### 5.2 Metrics

- train/val loss
- train/val residue accuracy over valid positions

### 5.3 Optimizer

- AdamW
- cosine LR schedule

## 6) Single-GPU operating mode

Default runtime profile:

- precision: `bf16-mixed`
- batch size: `1`
- gradient clipping: `1.0`
- gradient checkpointing: enabled
- max nodes: bounded by data crop

## 7) Training and inference interfaces

### 7.1 Train

`python scripts/train.py`

Must read config from `configs/config.yaml` and support overridden paths through Hydra.

### 7.2 Inference

`python scripts/inference.py --pdb <path_to_pdb> --out_fasta <path>`

Workflow:

1. Parse and featurize one structure.
2. Run forward pass.
3. Decode argmax or temperature sample.
4. Write FASTA.

## 8) Validation checklist

Before extending architecture, verify:

1. `python test_instantiate.py` works.
2. `python test_quick.py` runs forward pass.
3. `pytest -q` passes shape/collation tests.
4. `python scripts/pilot_run.py` runs on one real structure.

## 9) Planned v2 extensions

- Add sidechain coordinate auxiliary head (coordinate loss + sequence loss).
- Add random decoding order rather than fixed left-to-right.
- Add distance-noise augmentation scheduling.
- Add optional diffusion-style ligand-aware refinement head.
