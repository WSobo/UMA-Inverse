# SLURM job wrappers

`sbatch` wrappers for running UMA-Inverse on a SLURM HPC cluster. They are thin
shells around the Python entry points in `scripts/` and `uma-inverse …`; the
actual logic lives there. **Expect hardcoded cluster paths and `A5500`/`A100`
gres requests** — adapt the `#SBATCH` headers to your own partition and storage.

These are not part of the public API. To just run inference you need none of
them — see the top-level [README](../../README.md).

## Canonical v5 pipeline

The shipped model (`WSobo/UMA-Inverse`) is produced by this sequence:

| Stage | Script | Does |
|---|---|---|
| Fetch | `01a_fetch_data.sh` | Download PDBs from RCSB (`download_json_pdbs.py`) |
| Preprocess | `01c_preprocess_v5.sh` | Cache PDBs as `.pt` tensors (`preprocess_v5.py`) |
| Pilot | `02_pilot_run.sh` | 1-batch overfit sanity check |
| Train | `04a_v5_train_stage1.sh` → `04b_v5_train_stage2_ddp.sh` → `04c_v5_train_stage3_ddp.sh` | 3-stage curriculum (64 → 128 → 384 nodes) |
| Benchmark | `05h_benchmark_v5_full.sh`, `05i_benchmark_v5_interface.sh` | Test-split + interface-recovery metrics |

The `Makefile` targets (`make preprocess`, `make train-v5`, `make benchmark`, …)
wrap the same steps for convenience.

## Paper reproduction

The `preprint_*.sh` scripts regenerate the bioRxiv figures and statistics. They
are driven by the reproduction pipeline documented in
[`../paper/README.md`](../paper/README.md) (pocket-fixed designs, Boltz-2
cofolding, distal-signal analysis).

## Historical / superseded

Earlier architecture iterations (`*_v3_*`, `*_v4_*`, `02_pilot_v3.sh`,
`02_pilot_v4.sh`), `*_resume*.sh` restart wrappers, and one-off diagnostics
(`diag_ar_path.sh`, `probe_throughput.sh`, `pilot_64_*.sh`) are kept for
provenance. They are **not** needed to train or reproduce the v5 model.
