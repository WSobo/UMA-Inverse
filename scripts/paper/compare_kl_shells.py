"""Overlay per-distance-shell ligand-information KL across checkpoints.

Reads the per-checkpoint outputs produced by ``distal_kl_shift.py`` (one
``--out-dir`` per checkpoint, each holding ``per_position.parquet`` and
``distal_kl_summary.csv``) and overlays the per-shell mean KL(P_ligand ||
P_no-ligand) so the distogram effect is visible by distance to the ligand.

The intended comparison for v5 Phase A::

    baseline  = outputs/preprint/distal_kl/v4-init-ep6
    Run A     = outputs/preprint/distal_kl/runA-distogram-ep3   (λ=0.2)
    Control   = outputs/preprint/distal_kl/runA-control-ep3     (λ=0)

Reads from the parquet (not just the summary CSV) so it can recompute means on
a *common* set of (pdb, residue) positions across all checkpoints — otherwise a
PDB that one checkpoint happened to skip would bias that checkpoint's shell mean.

Usage::

    uv run python scripts/paper/compare_kl_shells.py \\
        --run baseline=outputs/preprint/distal_kl/v4-init-ep6 \\
        --run runA=outputs/preprint/distal_kl/runA-distogram-ep3 \\
        --run control=outputs/preprint/distal_kl/runA-control-ep3 \\
        --out-dir outputs/preprint/distal_kl/compare
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_kl")

# Matches DISTANCE_BINS in distal_kl_shift.py.
SHELLS: list[tuple[str, float, float]] = [
    ("0-5", 0.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-15", 10.0, 15.0),
    ("15-25", 15.0, 25.0),
    (">25", 25.0, float("inf")),
]


def _parse_runs(raw: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for entry in raw:
        if "=" not in entry:
            raise SystemExit(f"--run expects name=path, got {entry!r}")
        name, path = entry.split("=", 1)
        runs[name.strip()] = Path(path.strip())
    return runs


def _load(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "per_position.parquet"
    if not pq.exists():
        raise SystemExit(f"missing {pq} — run 05e_kl_shells.sh for this checkpoint first")
    df = pd.read_parquet(pq, columns=["pdb_id", "residue_idx", "dist_to_ligand", "kl_shift"])
    df = df[df["model"] == "uma-inverse-v3"] if "model" in df.columns else df
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True,
                    help="name=path/to/distal_kl/out-dir (repeatable; >=2).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/preprint/distal_kl/compare"))
    ap.add_argument("--no-common-set", action="store_true",
                    help="Don't restrict to positions present in every run.")
    args = ap.parse_args()

    runs = _parse_runs(args.run)
    if len(runs) < 2:
        raise SystemExit("need at least two --run entries to compare")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {name: _load(path) for name, path in runs.items()}
    for name, df in frames.items():
        logger.info("%-12s %7d positions across %4d PDBs",
                    name, len(df), df["pdb_id"].nunique())

    # ── Restrict to (pdb, residue) positions present in *every* run ──────────
    if not args.no_common_set:
        keysets = [set(map(tuple, df[["pdb_id", "residue_idx"]].to_numpy())) for df in frames.values()]
        common = set.intersection(*keysets)
        logger.info("common positions across all runs: %d", len(common))
        idx = pd.MultiIndex.from_tuples(sorted(common), names=["pdb_id", "residue_idx"])
        for name in frames:
            f = frames[name].set_index(["pdb_id", "residue_idx"])
            frames[name] = f.loc[f.index.intersection(idx)].reset_index()

    # ── Per-shell mean KL per run ────────────────────────────────────────────
    rows = []
    for label, lo, hi in SHELLS:
        rec: dict[str, object] = {"shell": label, "lo_A": lo,
                                  "hi_A": None if np.isinf(hi) else hi}
        n_ref = None
        for name, df in frames.items():
            m = (df["dist_to_ligand"] >= lo) & (df["dist_to_ligand"] < hi)
            sub = df[m]
            rec[f"{name}_mean_kl"] = float(sub["kl_shift"].mean()) if len(sub) else float("nan")
            rec[f"{name}_median_kl"] = float(sub["kl_shift"].median()) if len(sub) else float("nan")
            if n_ref is None:
                n_ref = len(sub)
        rec["n_residues"] = n_ref
        rows.append(rec)
    summary = pd.DataFrame(rows)

    out_csv = args.out_dir / "kl_shell_comparison.csv"
    summary.to_csv(out_csv, index=False)
    logger.info("wrote %s", out_csv)

    # ── Console table ────────────────────────────────────────────────────────
    names = list(frames.keys())
    pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:.4f}")
    cols = ["shell", "n_residues"] + [f"{n}_mean_kl" for n in names]
    print("\n=== mean KL(P_ligand || P_no-ligand) by distance shell ===")
    print(summary[cols].to_string(index=False))
    # pairwise deltas vs the first run (treated as the reference / baseline)
    ref = names[0]
    for n in names[1:]:
        summary[f"delta_{n}_vs_{ref}"] = summary[f"{n}_mean_kl"] - summary[f"{ref}_mean_kl"]
        print(f"\nΔ mean KL  {n} − {ref}:")
        for _, r in summary.iterrows():
            print(f"  {r['shell']:>6s}: {r[f'delta_{n}_vs_{ref}']:+.4f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(SHELLS))
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in names:
            ax.plot(x, summary[f"{name}_mean_kl"], marker="o", label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s[0]} Å" for s in SHELLS])
        ax.set_xlabel("Distance to nearest ligand atom")
        ax.set_ylabel("Mean KL(P_ligand ‖ P_no-ligand)  (nats)")
        ax.set_title("Ligand information by distance shell")
        ax.legend()
        fig.tight_layout()
        fig_path = args.out_dir / "kl_shell_comparison.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("wrote %s", fig_path)
    except Exception as e:  # pragma: no cover
        logger.warning("figure skipped: %s", e)

    summary.to_csv(out_csv, index=False)  # rewrite with delta columns


if __name__ == "__main__":
    main()
