# UMA-Inverse: Order of Operations (OOO)

To ensure maximum reproducibility and prevent HPC queuing issues, UMA-Inverse strictly follows a numbered Order of Operations (OOO). This matches the Struct2Seq-GNN workflow but is adapted for PyTorch Lightning and the dense PairMixer engine.

All bash execution scripts are located in `scripts/SLURM/`.

## 1. Data Procurement & Preparation

*   **`01a_fetch_data.sh`**
    *   **What it does:** Reads LigandMPNN's `train.json` / `valid.json` and fetches the raw PDB files from RCSB.
    *   **Why:** Secures structural input files locally so training doesn't depend on external network APIs.
*   **`01b_preprocess.sh`**
    *   **What it does:** Runs `scripts/preprocess.py` to parse raw PDBs through the LigandMPNN feature extractor and caches them as PyTorch `.pt` dictionary tensors in `data/processed/`.
    *   **Why:** Prevents massive CPU-bound PDB parsing bottlenecks during parallel GPU training. 

## 2. Model Validation Sandbox

*   **`02_pilot_run.sh`**
    *   **What it does:** Uses `overfit_batches=1` to lock the model onto a single complex. Disables masking to force pure sequence memorization.
    *   **Why:** Establishes peak VRAM high-water marks and scientifically validates that gradient flow is unblocked and tensors are mathematically sound before burning 96 hours of compute.

## 3. Scale-Up Training

*   **`03_train_model.sh`**
    *   **What it does:** Executes Multi-GPU DDP training via PyTorch Lightning with a curriculum-learning approach (stepping context window from 64 -> 128 -> 384 nodes).
    *   **Why:** Cheaper local geometry learning in Stage 1/2 saves massive compute before learning complex far-domain allostery in Stage 3. Uses Tensor Cores / bf16 out of the box.

## 4. Evaluation 

*   **`04_inference.sh`**
    *   **What it does:** Runs inverse-folding sequence generation for a given backbone via `scripts/inference.py`. Includes support for `--fixed_residues` constraints.
    *   **Why:** Calculates Native Sequence Recovery (NSR), applies chosen temperature scaling, and outputs the final designed `.fasta` files for wet-lab validation or AlphaFold forward-folding verification.
