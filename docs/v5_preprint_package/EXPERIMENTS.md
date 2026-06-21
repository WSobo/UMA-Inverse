# UMA-Inverse (v5) preprint — experiments & figures plan

Single-model paper: **UMA-Inverse = v5**. No v3/v4 versions shown; their components are
described as design choices. LigandMPNN/ProteinMPNN baselines are **published, fixed**
numbers on **fixed test splits** — so v5 is directly comparable just by running the same
protocols (no LigandMPNN re-run needed for the headline interface table).

Canonical checkpoint: `checkpoints/uma-inverse-v5.ckpt` (min-val-loss snapshot, promoted at
training-job exit = epoch 11). Until that exists, scripts auto-fall back to
`checkpoints/pairmixerinv-v5-stage3-nodes384-ddp2/epoch_snapshots/epoch-11.ckpt`.
Benchmark config: `configs/config.yaml` (smoke-validated: trunk loads 257/257, distogram
head dropped, teacher-forced recovery ≈ 0.64 per-PDB).

Training job chain (as of build):
`33890847 (primary, RUNNING)` → afterany → `34161619 (resume → epoch 24 + promote)` →
afterok → `34194485 (05h full-val)`, `34194486 (05i interface)`.

**Recovered data source.** The gitignored `outputs/` that was cleaned from this repo survives in the
old repo `../UMA-Inverse/`. Restored so far: LigandMPNN matched-K0/gibbs/val2000 recovery CSVs, the
pocket-fixed selection JSON, and LigandMPNN pocket/cofold baseline metrics. **LigandMPNN baselines do
not change** — reuse them; only the UMA-Inverse (v5) side needs re-running. Boltz-2 lives at
`../boltz2/`.

---

## Status legend
- ✅ **QUEUED** — sbatch'd with `--dependency=afterok:34161619`; runs automatically post-training.
- 🟡 **READY, NOT QUEUED** — script + inputs exist; one command away. Not auto-queued to avoid
  GPU contention / because it needs the canonical ckpt or a human decision.
- 🔴 **NEEDS SETUP** — missing input, stale path, or an upstream run must finish first.

---

## Experiments

### E1 — Teacher-forced full-val recovery + perplexity + ECE + per-AA + ligand ablation ✅ QUEUED
- **Job:** `34194485` (`05h_benchmark_v5_full.sh`), A5500, `afterok:34161619`.
- **Produces:** `outputs/benchmark/v5-best-full/` → per-PDB recovery, pooled recovery, perplexity,
  ECE, per-AA table, ligand-ablation Δ, temperature sweep {0.0, 0.1, 0.2}.
- **Fills:** abstract TF%, §Teacher-forced (TF/pool/ppl/ece/ablation), Table per-AA recovery.
- **Caveat:** v5 val set includes NA complexes → not 1:1 with any protein-only number; report on
  its own eval and say so.

### E2 — Interface recovery (LigandMPNN head-to-head), all 3 splits ✅ QUEUED
- **Job:** `34194486` (`05i_benchmark_v5_interface.sh`), A5500, `afterok:34161619`.
- **Protocol:** 10 AR samples, T=0.1, 5 Å sidechain cutoff, mean-of-per-PDB-medians.
- **Produces:** `outputs/benchmark/interface_recovery/v5-test_{small_molecule,metal,nucleotide}/`
  + `summarize_test_benchmarks.py` table vs LigandMPNN (0.633/0.775/0.505) & ProteinMPNN.
- **Fills:** abstract SM/ME/NU%, §Interface, Table iface, Fig 2/3.
- **v5-only:** nucleotide split is now in scope (NA routing). **The headline result.**

### E3 — Pocket-fixed design (pocket vs distal recovery + diversity) 🟡 READY, NOT QUEUED
- **Runs on the TEST SET:** `run_pocket_fixed_designs.py` already **defaults** to
  `data/raw/pdb_archive/test_{metal,small_molecule}`. The selection JSON just curates which test
  PDBs + precomputes ≤8 Å pocket residues.
- **Selection RESTORED** from `../UMA-Inverse/` → `outputs/preprint/pdb_selection_combined.json`
  (the v3 curated set: small_molecule + metal classes). No regeneration needed.
- **Run (UMA side):** `sbatch scripts/SLURM/preprint_uma_pocket_fixed.sh --ckpt checkpoints/uma-inverse-v5.ckpt --config configs/config.yaml --selection outputs/preprint/pdb_selection_combined.json --out-dir outputs/preprint/uma_pocket_fixed_v5`
- **LigandMPNN baseline:** restorable from `../UMA-Inverse/outputs/preprint/ligandmpnn_pocket_fixed`
  (LigandMPNN doesn't change — reuse, don't re-run). Ref metrics copied to
  `outputs/preprint/ligandmpnn_pocket_fixed_metrics_REF.csv`.
- **Note:** fix the stale `…/UMA-Inverse` (non-`-2`) `PROJ`/`#SBATCH --output` paths in
  `preprint_uma_pocket_fixed.sh` before submitting.
- **Fills:** §Pocket-fixed (POC%/DIST%), Fig 5.

### E4 — Boltz-2 cofolding of pocket-fixed designs 🟡 READY, NOT QUEUED (heavy)
- **Boltz-2** (not Boltz-1): scripts use `boltz predict` from the `…/computational/boltz2` env.
- **Scripts:** `preprint_boltz_cofold.sh` (+ `build_cofold_yamls.py`, `cofold_metrics.py`).
- **Depends on E3** designs; GPU, ~hours. **Highest-effort branch.**
- **LigandMPNN cofold baseline** restorable from `../UMA-Inverse/outputs/preprint/cofold*/ligandmpnn`
  + `cofold_metrics*.csv` (reuse — LigandMPNN unchanged).
- **Note:** fix stale `…/UMA-Inverse` paths in the boltz scripts first.
- **Fills:** §Pocket-fixed cofolding (conf/ipTM/ligand-ipTM), Fig 6.

### E5 — Ligand-distal signal (native log-prob vs distance shell) 🟡 READY, NOT QUEUED
- **Script:** `preprint_distal_kl_shift.sh` (env `KL_CKPT`, `KL_MODE=mechanism|outcome`, `KL_N`).
- **Run:** `sbatch --dependency=afterok:34161619 --export=ALL,KL_CKPT=checkpoints/uma-inverse-v5.ckpt,KL_CONFIG=configs/config.yaml,KL_MODE=outcome scripts/SLURM/preprint_distal_kl_shift.sh`
  (verify it accepts a `KL_CONFIG`; defaults to config_v3 — must override to config_v5).
- **Fills:** §Distal signal, Fig 7.

### E6 — Distogram probe (encoder geometry diagnostic) 🟡 READY, NOT QUEUED
- **Script:** `preprint_distogram_probe.sh` (env `PROBE_CKPT`, `PROBE_CONFIG`, `PROBE_N`).
- **Run:** `sbatch --dependency=afterok:34161619 --export=ALL,PROBE_CKPT=checkpoints/uma-inverse-v5.ckpt,PROBE_CONFIG=configs/config.yaml,PROBE_N=400 scripts/SLURM/preprint_distogram_probe.sh`
- **Note:** especially relevant for v5 — the distogram is now a *trained head*, not just a probe.
  Reports top-1 binned-distance accuracy (the in-training metric ≈ 0.93 for cross-check).
- **Fills:** §Distogram representation, distogram top-1 numbers throughout.

### E7 — Distogram ablation: Run A vs Control ⛔ DROPPED
- **Decision (2026-06-11):** out of scope. The distogram's benefit is argued by design rationale
  + the auxiliary head's high top-1 accuracy (E6), not a controlled on/off ablation. The Run A /
  Control warm-start checkpoints are only at epoch-00; training them out is not worth it.

### E8 — LigandMPNN K=0 matched head-to-head (paired, identical PDBs) ✅ DATA RESTORED
- **The normal LigandMPNN run on the same PDBs** for a paired comparison with UMA-Inverse —
  **already done in `../UMA-Inverse/`**, now restored:
  - `outputs/benchmark/ligandmpnn-matched-k0/per_pdb.csv` (492 PDBs; `pdb_id,recovery,n_residues`)
  - `outputs/benchmark/ligandmpnn-matched-gibbs/`, `ligandmpnn-val2000/` also restored.
- **Remaining:** run the UMA-Inverse v5 side at K=0 (single AR sample, T=0.1) on the same PDB set,
  then `scripts/paper/compare_k0_ligandmpnn.py` (paired t-test on the pdb_id intersection).
- **Note:** the interface table (E2) uses LigandMPNN's *published* numbers; this E8 paired run is
  the stronger reviewer-grade head-to-head on identical structures.

### E9 — Wallclock / throughput probe 🟡 READY, NOT QUEUED (minor)
- **Script:** `scripts/probe_inference_wallclock.py`. Optional efficiency footnote.

---

## Figures to show off

| Fig | Content | Experiment | Generator | Status |
|-----|---------|-----------|-----------|--------|
| **1** | Architecture schematic (PairMixer encoder + distogram head + ligand-attention decoder) | — | hand-drawn / TikZ | 🔴 new; the v5 story diagram (highlight distogram + ligand attention + NA routing) |
| **2** | Interface recovery bars vs LigandMPNN/ProteinMPNN, **3 splits incl. nucleotide** | E2 | `fig2_benchmark_bars.py` | 🟡 regen after E2 — **the headline** |
| **3** | Per-PDB interface recovery violins | E2 | `fig3_violins.py` | 🟡 regen after E2 |
| **4** | Training curves (val acc/loss) **+ distogram top-1 overlay** | logs | `fig4_training.py` | 🟡 regen from v5 `logs/csv/` (add distogram panel) |
| **5** | Pocket vs distal recovery + diversity | E3 | `fig5_pocket_distal.py` | 🔴 after E3 |
| **6** | Boltz-1 cofolding confidence/ipTM | E4 | `fig6_cofold.py` | 🔴 after E4 |
| **7** | Ligand-distal signal by shell **+ distogram-top1-by-shell** | E5/E6 | `fig7_distal_signal.py` | 🟡 after E5/E6 |

**Show-off priority** (capability-led framing, per the strategy):
1. **Fig 2** — interface recovery incl. the *new* nucleotide comparison (v5-only capability).
2. **Fig 1** — architecture, foregrounding the two novelties (distogram supervision, learned
   ligand attention) + NA routing.
3. **Fig 4 + Fig 7** — the distogram is structure-predictive (top-1 ≈ 0.93) and ligand signal
   reaches distal residues (the mean-pool→attention fix paying off).

---

## Order of operations after training finishes
1. (auto) E1, E2 run via `afterok:34161619` → primary quantitative tables + Figs 2–4.
2. Manually launch E5, E6 (one-liners above) once `uma-inverse-v5.ckpt` exists.
3. Run E3 (UMA pocket-fixed on test set; selection already restored) → E4 (Boltz-2 cofold) →
   Figs 5–6. Fix stale `…/UMA-Inverse` paths in those `preprint_*` scripts first.
4. E8: run UMA v5 K=0 on the matched PDBs, then `compare_k0_ligandmpnn.py` against the restored
   `ligandmpnn-matched-k0/per_pdb.csv` (paired t-test).
5. Fill every `\pending{…}` in `preprint.tex` (grep `pending`), regenerate figures, compile.
