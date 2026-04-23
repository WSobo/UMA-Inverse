"""Pure-function metrics for UMA-Inverse benchmarking.

Each function takes plain tensors or arrays and returns a small, scalar
or tensor-valued result — no I/O, no model calls. Keep them here so the
test suite can verify the arithmetic without spinning up PyTorch
Lightning.

All functions operate on LigandMPNN's 21-token alphabet
(``ACDEFGHIKLMNPQRSTVWY`` = 0..19, ``X`` = 20). Token 20 is never counted
as a valid prediction — we exclude it from recovery and perplexity by
default so the reported numbers match what LigandMPNN reports.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from src.utils.io import ID_TO_AA

# Convenience arrays used by multiple functions
_AA_LETTERS: list[str] = [ID_TO_AA[i] for i in range(21)]
_X_TOKEN: int = 20


# ─── Recovery ─────────────────────────────────────────────────────────────────


def recovery_rate(
    predicted: torch.Tensor,
    native: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Sequence recovery: fraction of masked positions where ``predicted == native``.

    Args:
        predicted: ``[L]`` or ``[B, L]`` int tensor of predicted token ids.
        native: Same shape — ground-truth token ids.
        mask: Optional bool mask of positions to include in the denominator.
            ``None`` → all positions count; ``X`` (token 20) is always excluded.

    Returns:
        Scalar float in ``[0, 1]``.
    """
    if predicted.shape != native.shape:
        raise ValueError(
            f"shape mismatch: predicted {predicted.shape} vs native {native.shape}"
        )
    valid = native != _X_TOKEN
    if mask is not None:
        valid = valid & mask.bool()
    if not valid.any():
        return 0.0
    correct = (predicted == native) & valid
    return float(correct.sum().item()) / float(valid.sum().item())


def per_aa_recovery(
    predicted: torch.Tensor,
    native: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Recovery broken down by native amino acid.

    Returns a dict ``{letter: recovery}`` for each AA that actually appears
    in ``native`` under ``mask``. Missing AAs are omitted (no NaNs). Useful
    for "model recovers W 82% of the time but K only 51%" statements.
    """
    valid = native != _X_TOKEN
    if mask is not None:
        valid = valid & mask.bool()

    out: dict[str, float] = {}
    for tok in range(20):  # 0..19 = real AAs
        letter = _AA_LETTERS[tok]
        is_this_aa = (native == tok) & valid
        n = int(is_this_aa.sum().item())
        if n == 0:
            continue
        hits = int(((predicted == native) & is_this_aa).sum().item())
        out[letter] = hits / n
    return out


# ─── Likelihood / perplexity ──────────────────────────────────────────────────


def perplexity(log_probs: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Mean-per-position perplexity: ``exp(-mean(log_probs))``.

    Conventionally reported in natural-log units (nats); multiply by
    ``1/ln(2)`` for bits if you prefer.

    Args:
        log_probs: ``[L]`` or ``[B, L]`` natural-log probabilities of the
            native token at each position.
        mask: Optional bool mask of included positions.

    Returns:
        Positive scalar. Lower is better; uniform guesses over 20 AAs give
        perplexity 20.
    """
    lp = log_probs if mask is None else log_probs[mask.bool()]
    if lp.numel() == 0:
        return float("nan")
    return float(torch.exp(-lp.mean()).item())


# ─── Confusion matrix ─────────────────────────────────────────────────────────


def confusion_matrix(
    predicted: torch.Tensor,
    native: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_classes: int = 20,
) -> np.ndarray:
    """Count matrix ``[num_classes, num_classes]``: rows = native, cols = predicted.

    Token 20 (``X``) is excluded from both axes by default so the matrix
    is strictly over the real-AA alphabet.
    """
    if num_classes > 21:
        raise ValueError(f"num_classes must be ≤ 21; got {num_classes}")
    valid = (native < num_classes) & (predicted < num_classes) & (native != _X_TOKEN)
    if mask is not None:
        valid = valid & mask.bool()

    native_flat = native[valid].to(torch.long).cpu().numpy()
    pred_flat = predicted[valid].to(torch.long).cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (native_flat, pred_flat), 1)
    return cm


def confusion_matrix_normalized(cm: np.ndarray, axis: str = "native") -> np.ndarray:
    """Row-normalised (``axis='native'``) or column-normalised confusion matrix.

    ``axis='native'`` → each row sums to 1 — answers "given a W residue,
    what distribution of predictions does the model produce?" (the common
    reading). ``axis='predicted'`` → each column sums to 1 — answers "when
    the model predicts W, what AA was it actually?"
    """
    if axis not in ("native", "predicted"):
        raise ValueError(f"axis must be 'native' or 'predicted'; got {axis!r}")
    denom = cm.sum(axis=1, keepdims=True) if axis == "native" else cm.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, cm / np.clip(denom, 1, None), 0.0)
    return out


# ─── Calibration ──────────────────────────────────────────────────────────────


@dataclass
class CalibrationBin:
    """One bin in a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_predicted_prob: float
    mean_accuracy: float


def calibration_bins(
    predicted_probs: torch.Tensor,
    correct: torch.Tensor,
    num_bins: int = 10,
) -> list[CalibrationBin]:
    """Bucket predictions by confidence and return accuracy per bucket.

    Args:
        predicted_probs: ``[N]`` — the probability the model assigned to
            the token it picked at each position.
        correct: ``[N]`` bool — whether the model's pick matched native.
        num_bins: Equal-width bins over ``[0, 1]``.

    Returns:
        List of :class:`CalibrationBin`, one per bucket. A perfectly
        calibrated model has ``mean_predicted_prob == mean_accuracy`` in
        every row.
    """
    if predicted_probs.shape != correct.shape:
        raise ValueError(
            f"shape mismatch: probs {predicted_probs.shape} vs correct {correct.shape}"
        )
    probs = predicted_probs.detach().cpu().numpy()
    hits = correct.detach().cpu().numpy().astype(bool)
    edges = np.linspace(0.0, 1.0, num_bins + 1)

    out: list[CalibrationBin] = []
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi if i < num_bins - 1 else probs <= hi)
        count = int(in_bin.sum())
        if count == 0:
            out.append(CalibrationBin(lower=lo, upper=hi, count=0,
                                      mean_predicted_prob=float("nan"),
                                      mean_accuracy=float("nan")))
            continue
        out.append(
            CalibrationBin(
                lower=float(lo),
                upper=float(hi),
                count=count,
                mean_predicted_prob=float(probs[in_bin].mean()),
                mean_accuracy=float(hits[in_bin].mean()),
            )
        )
    return out


def expected_calibration_error(bins: Sequence[CalibrationBin]) -> float:
    """Expected Calibration Error: weighted |predicted - accuracy| across bins.

    Standard summary number for reliability: lower = better calibrated.
    """
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")
    return float(
        sum(
            (b.count / total) * abs(b.mean_predicted_prob - b.mean_accuracy)
            for b in bins
            if b.count > 0
        )
    )


# ─── Entropy / diversity ──────────────────────────────────────────────────────


def entropy_per_position(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) per position.

    Args:
        probs: ``[L, K]`` or ``[B, L, K]`` — probability distributions.

    Returns:
        ``[L]`` or ``[B, L]`` entropy values. Zeros on degenerate (delta)
        distributions, ``log(K)`` on uniform.
    """
    safe = probs.clamp_min(1e-30)
    ent = -(safe * torch.log(safe)).sum(dim=-1)
    return ent


def hamming_diversity(sequences: Sequence[torch.Tensor]) -> float:
    """Mean pairwise Hamming distance (fraction of differing positions).

    Computed over all unordered sample pairs. Returns 0.0 for a single
    sequence. Used by the temperature sweep to quantify how much
    diversity higher temperatures actually buy.
    """
    n = len(sequences)
    if n < 2:
        return 0.0
    ref_len = sequences[0].numel()
    for idx, s in enumerate(sequences):
        if s.numel() != ref_len:
            raise ValueError(
                f"sequence {idx} has length {s.numel()}; expected {ref_len} "
                "(all sequences must have the same length)"
            )
    mat = torch.stack(list(sequences)).to(torch.long)  # [n, L]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = (mat[i] != mat[j]).float().mean().item()
            total += diff
            count += 1
    return total / count


# ─── AA composition ───────────────────────────────────────────────────────────


def aa_composition(
    sequences: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, float]:
    """Return the observed frequency of each amino acid.

    Args:
        sequences: ``[N, L]`` or ``[L]`` int token ids.
        mask: Optional bool mask selecting positions to count.

    Returns:
        ``{letter: frequency}`` summing to 1 over the 20 real AAs (X is
        excluded from the denominator as well as the numerator).
    """
    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)
    valid = sequences != _X_TOKEN
    if mask is not None:
        valid = valid & mask.bool()
    if not valid.any():
        return {letter: 0.0 for letter in _AA_LETTERS[:20]}
    counts: dict[str, int] = {letter: 0 for letter in _AA_LETTERS[:20]}
    flat = sequences[valid].cpu().tolist()
    for tok in flat:
        if 0 <= tok < 20:
            counts[_AA_LETTERS[tok]] += 1
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


# ─── Distance-to-ligand helper ────────────────────────────────────────────────


def residue_ligand_distances(
    residue_coords: torch.Tensor, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Return ``[L]`` min distance from each Cα to any ligand heavy atom.

    Returns a tensor of ``inf`` values when there are no ligand atoms.
    Used by the near-ligand/far breakdown.
    """
    if ligand_coords.numel() == 0:
        return torch.full(
            (residue_coords.shape[0],), float("inf"), device=residue_coords.device
        )
    dists = torch.cdist(residue_coords, ligand_coords)
    return dists.min(dim=-1).values
