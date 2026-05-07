# Paper reproduction scripts

This directory contains the scripts used to generate every figure and statistic in the UMA-Inverse bioRxiv preprint (link added on submission). They are kept here so the paper is fully reproducible from the published checkpoint and the LigandMPNN test-split JSONs.

These scripts are **not** required to use UMA-Inverse — for inference, see the top-level [README](../../README.md). They are also not part of the public API; expect rough edges and hardcoded HPC paths in places.

## Pipeline

The pocket-fixed redesign experiment (paper §3.4–§3.5) was produced in this order:

1. **PDB selection** — pick test-split PDBs that pass the hard filters (50 ≤ L ≤ 400, 5 ≤ pocket ≤ 30, distinct CCD codes for small-molecule split, IMAC-artifact filter for metal split).
   - `select_pdbs.py` — original 10 metal + 10 small-mol PDBs (with v2-friendly tiebreaker)
   - `select_extended_smallmol_pdbs.py` — +15 small-mol PDBs without the v2 tiebreaker (selection-bias scrub)

2. **Pocket-fixed designs** — run UMA-Inverse and LigandMPNN on the same PDBs with pocket residues fixed.
   - `run_pocket_fixed_designs.py` — UMA designs via `DesignConstraints.fix`
   - `build_ligandmpnn_inputs.py` + `../SLURM/preprint_ligandmpnn_pocket_fixed.sh` — LigandMPNN designs via `--fixed_residues_multi`

3. **Recovery + diversity metrics** — score each method's distal designs.
   - `compute_pocket_fixed_metrics.py` — per-PDB distal recovery, distal Hamming diversity, AA-frequency tables
   - `pocket_fixed_stats.py` — paired Wilcoxon + bootstrap 95% CIs

4. **Boltz-2 cofold harness** — cofold a sample of designs against the native ligand.
   - `build_boltz_yaml.py` / `build_cofold_yamls.py` — emit Boltz-2 YAML inputs (single + multi-chain; numeric CCDs always quoted)
   - `../SLURM/preprint_boltz_cofold*.sh` — `boltz predict` invocations
   - `cofold_metrics.py` — extract ipTM, pLDDT, pocket Cα RMSD, ligand-pose RMSD, scaffold RMSD, predicted affinity from cofold outputs

5. **Aggregation + figure generation** — bring the original + extended + rerun cofolds together and regenerate figures.
   - `finalize_combined_metrics.py` — runs steps 3 + 4 over the combined N=25 selection, emits `outputs/preprint/cofold_metrics.csv`, regenerates figs 5/6

6. **Test 4 correlation analysis** — per-PDB confidence vs cofold quality, the discriminating test in §3.4.
   - `distal_signal_analysis.py` — runs Tests 1 (between-method ranking corr), 2 (within-PDB AA agreement), 3 (pocket size vs diversity), 4 (per-PDB confidence vs cofold ipTM/pLDDT/RMSD/affinity)

7. **Figure scripts** — matplotlib code, source CSVs in `outputs/preprint/`.
   - `figures/fig2_benchmark_bars.py` — standard interface-recovery bars
   - `figures/fig3_violins.py` — per-PDB recovery violins
   - `figures/fig4_training.py` — curriculum training curves
   - `figures/fig5_pocket_distal.py` — pocket-fixed distal-diversity violins + scatter
   - `figures/fig6_cofold.py` — Boltz-2 cofold metrics

Figure 1 (architecture diagram) is generated from a standalone TikZ source kept with the preprint working files (outside this repo).

## Outputs

All generated CSVs and PDFs land under `outputs/preprint/` (gitignored). The canonical CSVs the paper depends on:

- `outputs/preprint/pocket_fixed_metrics.csv` — per-sample distal recovery + diversity
- `outputs/preprint/pocket_fixed_summary.csv` — per-PDB-per-method aggregate
- `outputs/preprint/pocket_fixed_aa_freq.csv` — distal AA-frequency tables (used by Test 2)
- `outputs/preprint/cofold_metrics.csv` — Boltz-2 cofold metrics (used by Test 4 + fig 6)
- `outputs/preprint/pdb_selection_combined.json` — final N=25 small-mol + 10 metal selection

## Running on a different cluster

The scripts assume:
- LigandMPNN's `run.py` is on `PYTHONPATH` (or a sibling install at `LigandMPNN/`)
- Boltz-2 is installed and reachable via `boltz predict` in a `boltz` micromamba env (per the [Boltz-2 release](https://github.com/jwohlwend/boltz))
- SLURM with a GPU partition (the SLURM wrappers in `../SLURM/` target an `A5500` gres but adapt to any modern NVIDIA GPU with ≥ 24 GB)

For a non-SLURM environment, the `.sh` wrappers can be replaced with bare invocations of the underlying Python scripts.
