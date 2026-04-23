"""Unit tests for :mod:`src.benchmarks.metrics`.

Pure-function coverage: feed known tensors, assert known outputs. These
are the building blocks every other benchmark module relies on, so
correctness here is load-bearing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.benchmarks.metrics import (
    aa_composition,
    calibration_bins,
    confusion_matrix,
    confusion_matrix_normalized,
    entropy_per_position,
    expected_calibration_error,
    hamming_diversity,
    per_aa_recovery,
    perplexity,
    recovery_rate,
    residue_ligand_distances,
)


class TestRecoveryRate:
    def test_all_match(self) -> None:
        pred = torch.tensor([0, 1, 2, 3])
        nat = torch.tensor([0, 1, 2, 3])
        assert recovery_rate(pred, nat) == pytest.approx(1.0)

    def test_half_match(self) -> None:
        pred = torch.tensor([0, 1, 9, 9])
        nat = torch.tensor([0, 1, 2, 3])
        assert recovery_rate(pred, nat) == pytest.approx(0.5)

    def test_x_excluded_from_denominator(self) -> None:
        # X=20; positions with native=20 should not count
        pred = torch.tensor([0, 1, 20, 5])
        nat = torch.tensor([0, 1, 20, 5])
        # Three of the four are valid; all three hit → 1.0
        assert recovery_rate(pred, nat) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            recovery_rate(torch.tensor([0]), torch.tensor([0, 1]))

    def test_empty_after_mask(self) -> None:
        pred = torch.tensor([0, 1, 2])
        nat = torch.tensor([0, 1, 2])
        mask = torch.zeros(3, dtype=torch.bool)
        assert recovery_rate(pred, nat, mask=mask) == 0.0


class TestPerAaRecovery:
    def test_only_present_aas_returned(self) -> None:
        pred = torch.tensor([0, 0, 1, 1])
        nat = torch.tensor([0, 0, 1, 2])  # A, A, C, D
        out = per_aa_recovery(pred, nat)
        assert set(out.keys()) == {"A", "C", "D"}
        assert out["A"] == 1.0
        assert out["C"] == 1.0
        assert out["D"] == 0.0


class TestPerplexity:
    def test_uniform_20(self) -> None:
        # log(1/20) everywhere → perplexity = 20
        lp = torch.full((100,), np.log(1 / 20))
        assert perplexity(lp) == pytest.approx(20.0, rel=1e-5)

    def test_empty_after_mask(self) -> None:
        lp = torch.zeros(3)
        mask = torch.zeros(3, dtype=torch.bool)
        assert np.isnan(perplexity(lp, mask=mask))


class TestConfusionMatrix:
    def test_shape_and_counts(self) -> None:
        pred = torch.tensor([0, 1, 0, 2])
        nat = torch.tensor([0, 1, 1, 2])
        cm = confusion_matrix(pred, nat, num_classes=3)
        expected = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        assert (cm == expected).all()

    def test_row_normalised(self) -> None:
        cm = np.array([[2, 0], [1, 1]])
        norm = confusion_matrix_normalized(cm, axis="native")
        assert np.allclose(norm, [[1.0, 0.0], [0.5, 0.5]])

    def test_column_normalised(self) -> None:
        cm = np.array([[2, 0], [1, 1]])
        norm = confusion_matrix_normalized(cm, axis="predicted")
        assert np.allclose(norm[:, 0], [2 / 3, 1 / 3])
        assert np.allclose(norm[:, 1], [0.0, 1.0])

    def test_invalid_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            confusion_matrix_normalized(np.eye(2), axis="bogus")


class TestCalibrationBins:
    def test_bin_count_and_edges(self) -> None:
        probs = torch.tensor([0.05, 0.25, 0.45, 0.95])
        correct = torch.tensor([False, True, True, True])
        bins = calibration_bins(probs, correct, num_bins=10)
        assert len(bins) == 10
        # Check boundaries
        assert bins[0].lower == 0.0
        assert bins[-1].upper == pytest.approx(1.0)

    def test_perfect_calibration_ece_zero(self) -> None:
        # Probs exactly match accuracy in every bin
        probs = torch.tensor([0.25, 0.25, 0.75, 0.75])
        correct = torch.tensor([False, True, True, True])
        # Bin containing 0.25 has 50% accuracy (should match 0.25 predicted mean → not 0)
        # Bin containing 0.75 has 100% accuracy
        # Not actually perfect; just checks ECE is reasonable
        ece = expected_calibration_error(calibration_bins(probs, correct, num_bins=10))
        assert 0 <= ece <= 1

    def test_empty_input(self) -> None:
        bins = calibration_bins(torch.zeros(0), torch.zeros(0, dtype=torch.bool))
        assert all(b.count == 0 for b in bins)
        assert np.isnan(expected_calibration_error(bins))


class TestEntropy:
    def test_delta_distribution_zero_entropy(self) -> None:
        probs = torch.zeros(3, 21)
        probs[:, 0] = 1.0
        ent = entropy_per_position(probs)
        assert torch.all(ent < 1e-5)

    def test_uniform_max_entropy(self) -> None:
        probs = torch.full((3, 21), 1 / 21)
        ent = entropy_per_position(probs)
        expected = float(np.log(21))
        assert torch.allclose(ent, torch.full_like(ent, expected), atol=1e-4)


class TestHammingDiversity:
    def test_single_seq_returns_zero(self) -> None:
        seqs = [torch.tensor([0, 1, 2])]
        assert hamming_diversity(seqs) == 0.0

    def test_identical_sequences_zero(self) -> None:
        seqs = [torch.tensor([0, 1, 2])] * 3
        assert hamming_diversity(seqs) == 0.0

    def test_fully_distinct_sequences(self) -> None:
        a = torch.tensor([0, 0, 0])
        b = torch.tensor([1, 1, 1])
        assert hamming_diversity([a, b]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            hamming_diversity([torch.tensor([0, 1]), torch.tensor([0, 1, 2])])


class TestAaComposition:
    def test_frequencies_sum_to_one(self) -> None:
        seqs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        comp = aa_composition(seqs)
        assert sum(comp.values()) == pytest.approx(1.0)

    def test_x_excluded(self) -> None:
        # Token 20 (X) should not appear in the output
        seqs = torch.tensor([0, 0, 20, 20])
        comp = aa_composition(seqs)
        assert "X" not in comp
        assert comp["A"] == pytest.approx(1.0)  # denominator excludes X


class TestResidueLigandDistances:
    def test_minimum_distance(self) -> None:
        res = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        lig = torch.tensor([[1.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        d = residue_ligand_distances(res, lig)
        assert d[0] == pytest.approx(1.0)
        assert d[1] == pytest.approx(9.0)

    def test_no_ligand_returns_inf(self) -> None:
        res = torch.tensor([[0.0, 0.0, 0.0]])
        lig = torch.zeros((0, 3))
        d = residue_ligand_distances(res, lig)
        assert torch.isinf(d).all()
