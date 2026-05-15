"""Distogram linear probe — does v3's Z_ij actually encode geometry?

The v3 retro question we couldn't answer from training-loss curves alone:
is the dense pair tensor learning a useful geometric representation, or is
the encoder structurally inert and the model leaning on local features
through the decoder? A linear probe on Cβ-Cβ distogram bins gives a direct
read.

Method:
    1. Load the v3 final checkpoint and freeze the trunk.
    2. For each PDB in a val sample, extract the encoder's final residue-
       residue pair tensor ``z[:L, :L, :pair_dim]`` and the ground-truth
       Cβ-Cβ pairwise distances (Cβ derived from N/Cα/C via the LigandMPNN
       formula, matching what the model itself sees at training time).
    3. Train a single ``Linear(pair_dim, n_bins)`` head with cross-entropy
       over 38 AF3-style distance bins (3.15–50.75 Å, 1.25 Å width).
    4. Evaluate on a held-out PDB split: top-1, top-3, neighbor-bin
       (top-1 ± 1 bin), expected-distance MAE, ECE.

Outcome thresholds (committed before reading numbers):
    top-1 > 0.85          → encoder encodes geometry well; v4 bottleneck
                            is decoder/data (prioritize Boltz-pretraining
                            and decoder upgrade).
    top-1 in [0.60, 0.85] → encoder encodes geometry partially; v4 wins
                            from feature density (AF3 pos enc, AtomFlow,
                            token_bonds).
    top-1 < 0.60          → Z_ij is not carrying geometry; v4 needs an
                            encoder rethink before features/decoder.

Usage:
    uv run python scripts/paper/distogram_probe.py \\
        --uma-ckpt checkpoints/uma-inverse-v3.ckpt \\
        --pdb-list outputs/preprint/distogram_probe/pdb_list.txt \\
        --out-dir outputs/preprint/distogram_probe
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ligandmpnn_bridge import resolve_pdb_path  # noqa: E402
from src.data.pdb_parser import parse_pdb  # noqa: E402
from src.inference.session import InferenceSession  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("distogram_probe")


# ─── Distance binning (AF3-template style) ────────────────────────────────────

BIN_LO = 3.15
BIN_HI = 50.75
N_BINS = 38
BIN_WIDTH = (BIN_HI - BIN_LO) / N_BINS  # 1.25 Å


def _bin_distances(dists: Tensor) -> Tensor:
    """Bin a [L, L] distance matrix into N_BINS classes.

    Bins 0..N_BINS-1 evenly span [BIN_LO, BIN_HI]. Distances < BIN_LO clamp
    to bin 0; distances > BIN_HI clamp to bin N_BINS-1. Returns long.
    """
    idx = ((dists - BIN_LO) / BIN_WIDTH).long()
    return idx.clamp(min=0, max=N_BINS - 1)


def _bin_centers(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Center distance for each bin, used for expected-distance MAE."""
    return torch.linspace(
        BIN_LO + BIN_WIDTH / 2.0,
        BIN_HI - BIN_WIDTH / 2.0,
        N_BINS,
        device=device,
        dtype=dtype,
    )


# ─── Cβ derivation (matches the model) ────────────────────────────────────────


def _derive_cb(bb: Tensor) -> Tensor:
    """Virtual Cβ from N/Cα/C. Same constants as uma_inverse.py:421."""
    n  = bb[..., 0, :]
    ca = bb[..., 1, :]
    c  = bb[..., 2, :]
    b = ca - n
    c_vec = c - ca
    a = torch.linalg.cross(b, c_vec, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


# ─── PDB list resolution (shared with distal_kl_shift) ────────────────────────


def _resolve_pdb_paths(
    pdb_list_path: Path,
    pdb_dirs: list[Path],
    max_residues: int | None,
) -> list[Path]:
    raw = [
        s.strip()
        for s in pdb_list_path.read_text().splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]
    paths: list[Path] = []
    for entry in raw:
        as_path = Path(entry)
        if as_path.exists() and as_path.suffix == ".pdb":
            paths.append(as_path)
            continue
        found = None
        for pdb_dir in pdb_dirs:
            cand = resolve_pdb_path(str(pdb_dir), entry)
            if cand is not None:
                found = Path(cand)
                break
        if found is None:
            logger.warning("could not resolve PDB %s in any of %s", entry, pdb_dirs)
            continue
        paths.append(found)

    if max_residues is not None:
        kept: list[Path] = []
        for p in paths:
            try:
                parsed = parse_pdb(str(p))
                n = int(parsed["mask"].sum().item())
            except Exception:
                continue
            if n <= max_residues:
                kept.append(p)
        paths = kept
    return paths


# ─── Z_ij + distogram extractor ───────────────────────────────────────────────


@torch.no_grad()
def _extract_pair_features(
    session: InferenceSession,
    pdb_path: Path,
) -> tuple[Tensor, Tensor] | None:
    """Return (z_res, dist_bins) for one PDB, or None on failure.

    z_res:     [L, L, pair_dim], float32, on session.device.
    dist_bins: [L, L], long, on session.device.
    """
    try:
        ctx = session.load_structure(pdb_path, mask_ligand=False, max_total_nodes=5000)
    except Exception as exc:
        logger.warning("skipping %s (load failed: %s)", pdb_path.stem, exc)
        return None

    L = ctx.residue_count
    if L < 8:
        return None  # too small to give meaningful pair statistics

    z_res = ctx.z[0, :L, :L, :].float()

    if ctx.residue_backbone_coords is None:
        # Without backbone coords we'd have to use the residue anchor
        # directly; for v3 this branch is unreachable because every
        # v3 config sets pair_distance_atoms*='backbone_full*', which
        # forces backbone coords to be present.
        logger.warning(
            "skipping %s (no residue_backbone_coords; non-v3 ckpt?)", pdb_path.stem
        )
        return None
    bb = ctx.residue_backbone_coords[0].float()  # [L, 4, 3]
    cb = _derive_cb(bb)                          # [L, 3]
    dists = torch.cdist(cb, cb)                  # [L, L]
    dist_bins = _bin_distances(dists)            # [L, L] long

    return z_res, dist_bins


# ─── Linear probe ─────────────────────────────────────────────────────────────


class DistogramHead(nn.Module):
    """Single linear layer pair_dim -> N_BINS, no bias regularization.

    Kept deliberately tiny — the point is to measure how much geometry the
    frozen trunk has *already* encoded, not to recover geometry from a deep
    MLP probe.
    """

    def __init__(self, pair_dim: int, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.proj = nn.Linear(pair_dim, n_bins)

    def forward(self, z: Tensor) -> Tensor:
        return self.proj(z)


def _sample_pair_indices(
    L: int,
    n_samples: int,
    *,
    min_seq_sep: int = 2,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample (i, j) pairs with |i-j| >= min_seq_sep. Returns row/col tensors."""
    # Rejection-sample on CPU; cheap because L^2 dominates the bottleneck
    # only at large L and we cap n_samples anyway.
    out_i: list[int] = []
    out_j: list[int] = []
    needed = n_samples
    while needed > 0:
        i = torch.randint(0, L, (needed * 2,), generator=rng)
        j = torch.randint(0, L, (needed * 2,), generator=rng)
        keep = (i - j).abs() >= min_seq_sep
        i = i[keep][:needed].tolist()
        j = j[keep][:needed].tolist()
        out_i.extend(i)
        out_j.extend(j)
        needed = n_samples - len(out_i)
    return torch.tensor(out_i[:n_samples]), torch.tensor(out_j[:n_samples])


def _train_probe(
    head: DistogramHead,
    train_samples: list[tuple[Tensor, Tensor]],
    *,
    pairs_per_pdb: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 0,
) -> list[float]:
    """Train the linear head on (z_res, dist_bins) tuples. Returns per-epoch loss."""
    head.train()
    optim = torch.optim.Adam(head.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    losses: list[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        running_n = 0
        # Single pass through PDBs per epoch — shuffle order.
        order = torch.randperm(len(train_samples), generator=rng).tolist()
        # Aggregate pair samples into mini-batches that span PDBs.
        buf_z: list[Tensor] = []
        buf_t: list[Tensor] = []
        for pdb_idx in order:
            z_res, dist_bins = train_samples[pdb_idx]
            L = z_res.shape[0]
            n_sample = min(pairs_per_pdb, L * L)
            ii, jj = _sample_pair_indices(L, n_sample, min_seq_sep=2, rng=rng)
            buf_z.append(z_res[ii, jj])           # [n_sample, pair_dim]
            buf_t.append(dist_bins[ii, jj])       # [n_sample]
            total = sum(b.shape[0] for b in buf_z)
            if total >= batch_size:
                z_batch = torch.cat(buf_z, dim=0).to(device)
                t_batch = torch.cat(buf_t, dim=0).to(device)
                buf_z, buf_t = [], []
                logits = head(z_batch)
                loss = nn.functional.cross_entropy(logits, t_batch)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                running_loss += float(loss.item()) * t_batch.shape[0]
                running_n += t_batch.shape[0]

        # Flush remaining
        if buf_z:
            z_batch = torch.cat(buf_z, dim=0).to(device)
            t_batch = torch.cat(buf_t, dim=0).to(device)
            logits = head(z_batch)
            loss = nn.functional.cross_entropy(logits, t_batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running_loss += float(loss.item()) * t_batch.shape[0]
            running_n += t_batch.shape[0]

        avg = running_loss / max(running_n, 1)
        losses.append(avg)
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, avg)

    return losses


@torch.no_grad()
def _eval_probe(
    head: DistogramHead,
    test_samples: list[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Compute top-1, top-3, neighbor-bin, expected-distance MAE, ECE."""
    head.eval()
    centers = _bin_centers(device, torch.float32)

    n_total = 0
    n_top1 = 0
    n_top3 = 0
    n_neighbor = 0
    mae_sum = 0.0
    ece_buckets = torch.zeros(10, device=device)
    ece_correct = torch.zeros(10, device=device)
    ece_counts = torch.zeros(10, device=device)

    # Per-bin recall for diagnostics (was the probe just predicting the
    # majority class?)
    per_bin_correct = torch.zeros(N_BINS, device=device)
    per_bin_total = torch.zeros(N_BINS, device=device)

    for z_res, dist_bins in test_samples:
        L = z_res.shape[0]
        # Mask diagonal and adjacent (|i-j|<2).
        ii, jj = torch.meshgrid(
            torch.arange(L), torch.arange(L), indexing="ij",
        )
        keep = (ii - jj).abs() >= 2
        z_flat = z_res[keep].to(device)         # [P, pair_dim]
        t_flat = dist_bins[keep].to(device)     # [P]

        logits = head(z_flat)                   # [P, N_BINS]
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)

        n = t_flat.shape[0]
        n_total += n
        n_top1 += int((pred == t_flat).sum().item())

        top3 = probs.topk(3, dim=-1).indices
        n_top3 += int((top3 == t_flat.unsqueeze(-1)).any(dim=-1).sum().item())
        n_neighbor += int(((pred - t_flat).abs() <= 1).sum().item())

        exp_dist = (probs * centers).sum(dim=-1)
        true_dist = centers[t_flat]
        mae_sum += float((exp_dist - true_dist).abs().sum().item())

        max_prob, _ = probs.max(dim=-1)
        buckets = (max_prob * 10).clamp(0, 9).long()
        correct = (pred == t_flat).float()
        for b in range(10):
            m = buckets == b
            if m.any():
                ece_correct[b] += correct[m].sum()
                ece_buckets[b] += max_prob[m].sum()
                ece_counts[b] += m.sum()

        for b in range(N_BINS):
            m = t_flat == b
            if m.any():
                per_bin_total[b] += m.sum()
                per_bin_correct[b] += (pred[m] == b).sum()

    # ECE = sum_b (N_b / N) * |acc_b - conf_b|
    valid = ece_counts > 0
    acc_b = torch.zeros_like(ece_counts)
    conf_b = torch.zeros_like(ece_counts)
    acc_b[valid] = ece_correct[valid] / ece_counts[valid]
    conf_b[valid] = ece_buckets[valid] / ece_counts[valid]
    weights = ece_counts / ece_counts.sum().clamp_min(1)
    ece = float((weights * (acc_b - conf_b).abs()).sum().item())

    per_bin_recall = (per_bin_correct / per_bin_total.clamp_min(1)).cpu().numpy()

    return {
        "n_pairs": int(n_total),
        "top1": n_top1 / max(n_total, 1),
        "top3": n_top3 / max(n_total, 1),
        "neighbor_top1": n_neighbor / max(n_total, 1),
        "expected_dist_mae_A": mae_sum / max(n_total, 1),
        "ece": ece,
        "per_bin_recall": per_bin_recall.tolist(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uma-ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "uma-inverse-v3.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config_v3.yaml",
    )
    parser.add_argument("--pdb-list", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, action="append", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprint" / "distogram_probe",
    )
    parser.add_argument("--max-residues", type=int, default=359)
    parser.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of PDBs used to fit the probe; rest are held out.",
    )
    parser.add_argument("--pairs-per-pdb", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.pdb_dir is None:
        archive = PROJECT_ROOT / "data" / "raw" / "pdb_archive"
        args.pdb_dir = [
            archive,
            archive / "test_metal",
            archive / "test_small_molecule",
        ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    max_residues = args.max_residues if args.max_residues > 0 else None

    pdb_paths = _resolve_pdb_paths(args.pdb_list, args.pdb_dir, max_residues)
    if not pdb_paths:
        logger.error("no PDBs resolved")
        return
    logger.info("resolved %d PDBs", len(pdb_paths))

    if not args.uma_ckpt.exists():
        logger.error("UMA ckpt not found: %s", args.uma_ckpt)
        return
    session = InferenceSession.from_checkpoint(
        config_path=args.config, checkpoint=args.uma_ckpt
    )
    device = session.device
    pair_dim = int(session.model.pair_dim)
    logger.info("loaded v3 trunk (pair_dim=%d) on %s", pair_dim, device)

    # Extract once — keeps memory bounded vs. running encoder every epoch.
    # Each tuple is (L, L, pair_dim) fp32 on CPU; for ~1000 PDBs at L<=359
    # this peaks around ~60 GB which fits in node RAM (32 GB allocated may
    # be tight; the SLURM wrapper requests 64 GB to be safe).
    cached: list[tuple[Tensor, Tensor]] = []
    for idx, pdb_path in enumerate(pdb_paths, 1):
        logger.info("extract %d/%d: %s", idx, len(pdb_paths), pdb_path.stem)
        result = _extract_pair_features(session, pdb_path)
        if result is None:
            continue
        z_res, dist_bins = result
        cached.append((z_res.cpu(), dist_bins.cpu()))
    logger.info("cached %d/%d PDBs successfully", len(cached), len(pdb_paths))
    if not cached:
        logger.error("no usable PDBs after extraction")
        return

    # Free the trunk's GPU footprint before training the head — keeps GPU
    # memory clean for the head's optimizer state.
    del session
    torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cached))
    split = int(len(cached) * args.train_frac)
    train_samples = [cached[i] for i in perm[:split]]
    test_samples = [cached[i] for i in perm[split:]]
    logger.info(
        "split: %d train PDBs, %d test PDBs", len(train_samples), len(test_samples)
    )

    head = DistogramHead(pair_dim=pair_dim, n_bins=N_BINS).to(device)
    losses = _train_probe(
        head,
        train_samples,
        pairs_per_pdb=args.pairs_per_pdb,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    metrics = _eval_probe(head, test_samples, device=device)

    # Verdict against the pre-committed thresholds.
    top1 = metrics["top1"]
    if top1 > 0.85:
        verdict = "encoder_strong"
    elif top1 >= 0.60:
        verdict = "encoder_partial"
    else:
        verdict = "encoder_weak"
    metrics["verdict"] = verdict

    out_json = args.out_dir / "distogram_probe_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "ckpt": str(args.uma_ckpt),
                "config": str(args.config),
                "n_train_pdbs": len(train_samples),
                "n_test_pdbs": len(test_samples),
                "pairs_per_pdb": args.pairs_per_pdb,
                "epochs": args.epochs,
                "lr": args.lr,
                "n_bins": N_BINS,
                "bin_lo_A": BIN_LO,
                "bin_hi_A": BIN_HI,
                "bin_width_A": BIN_WIDTH,
                "train_losses": losses,
                **metrics,
            },
            indent=2,
        )
    )
    pd.DataFrame(
        {
            "bin_idx": list(range(N_BINS)),
            "bin_center_A": [BIN_LO + BIN_WIDTH * (b + 0.5) for b in range(N_BINS)],
            "recall": metrics["per_bin_recall"],
        }
    ).to_csv(args.out_dir / "per_bin_recall.csv", index=False)

    logger.info("verdict=%s  top1=%.3f  top3=%.3f  neighbor=%.3f  MAE=%.2f Å  ECE=%.3f",
        verdict, metrics["top1"], metrics["top3"], metrics["neighbor_top1"],
        metrics["expected_dist_mae_A"], metrics["ece"],
    )
    logger.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
