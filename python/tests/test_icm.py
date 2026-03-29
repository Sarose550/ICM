"""Tests for the ICM Python bindings."""

import numpy as np
import pytest

import icm


@pytest.fixture(scope="session", autouse=True)
def initialize():
    """Initialize ICM library once for all tests."""
    icm.init()


class TestEquityBasic:
    """Basic equity computation tests."""

    def test_n3_equal_stacks(self):
        """Three equal stacks should produce equal equities."""
        eq = icm.equity([1000, 1000, 1000], [0.5, 0.3, 0.2])
        # With equal stacks, all equities should be equal
        np.testing.assert_allclose(eq[0], eq[1], rtol=1e-6)
        np.testing.assert_allclose(eq[1], eq[2], rtol=1e-6)

    def test_n3_unequal_stacks(self):
        """Larger stacks should produce higher equities."""
        eq = icm.equity([1000, 2000, 3000], [0.5, 0.3, 0.2])
        assert eq[2] > eq[1] > eq[0], (
            f"Expected eq[2] > eq[1] > eq[0], got {eq}"
        )

    def test_n5_equal_stacks(self):
        """Five equal stacks with 3 payouts."""
        stacks = [1000] * 5
        payouts = [0.5, 0.3, 0.2]
        eq = icm.equity(stacks, payouts)
        # All equities should be equal
        for i in range(1, 5):
            np.testing.assert_allclose(eq[i], eq[0], rtol=1e-6)

    def test_n5_unequal(self):
        """Five players with varying stacks."""
        stacks = [500, 1000, 1500, 2000, 5000]
        payouts = [0.40, 0.25, 0.18, 0.10, 0.07]
        eq = icm.equity(stacks, payouts)
        # Equities should be monotonically increasing with stack size
        for i in range(len(stacks) - 1):
            assert eq[i] < eq[i + 1], (
                f"Expected eq[{i}] < eq[{i+1}], got {eq[i]:.6f} >= {eq[i+1]:.6f}"
            )


class TestEquitySum:
    """Verify that equities sum to the total payout."""

    def test_sum_n3(self):
        """Equities should sum to sum(payouts) for n=3."""
        payouts = [0.5, 0.3, 0.2]
        eq = icm.equity([1000, 2000, 3000], payouts)
        np.testing.assert_allclose(eq.sum(), sum(payouts), rtol=1e-8)

    def test_sum_n5(self):
        """Equities should sum to sum(payouts) for n=5."""
        payouts = [0.40, 0.25, 0.18, 0.10, 0.07]
        eq = icm.equity([500, 1000, 1500, 2000, 5000], payouts)
        np.testing.assert_allclose(eq.sum(), sum(payouts), rtol=1e-8)

    def test_sum_k_less_than_n(self):
        """With k < n, equities still sum to sum(payouts)."""
        stacks = [1000, 2000, 3000, 4000, 5000]
        payouts = [0.5, 0.3]  # k=2, n=5
        eq = icm.equity(stacks, payouts)
        np.testing.assert_allclose(eq.sum(), sum(payouts), rtol=1e-8)


class TestV1Linear:
    """Test V1 (linear payout) against exact closed-form values.

    For payout = [1.0] (V1), the equity of player i is exactly S_i / sum(S).
    """

    def test_v1_n3(self):
        stacks = [1000, 2000, 3000]
        eq = icm.equity(stacks, [1.0])
        total = sum(stacks)
        expected = np.array([s / total for s in stacks])
        np.testing.assert_allclose(eq, expected, rtol=1e-10)

    def test_v1_n5(self):
        stacks = [100, 200, 300, 400, 500]
        eq = icm.equity(stacks, [1.0])
        total = sum(stacks)
        expected = np.array([s / total for s in stacks])
        np.testing.assert_allclose(eq, expected, rtol=1e-10)

    def test_v1_n10(self):
        stacks = [100 * (i + 1) for i in range(10)]
        eq = icm.equity(stacks, [1.0])
        total = sum(stacks)
        expected = np.array([s / total for s in stacks])
        np.testing.assert_allclose(eq, expected, rtol=1e-10)


class TestSubset:
    """Test equity_subset computation."""

    def test_subset_matches_full(self):
        """Subset equities should match full computation at target indices."""
        stacks = [1000, 2000, 3000, 4000, 5000]
        payouts = [0.5, 0.3, 0.2]
        targets = [0, 2, 4]

        eq_full = icm.equity(stacks, payouts)
        eq_sub = icm.equity_subset(stacks, payouts, targets)

        for t in targets:
            np.testing.assert_allclose(
                eq_sub[t], eq_full[t], rtol=1e-10,
                err_msg=f"Mismatch at target index {t}"
            )

    def test_subset_single_target(self):
        """Single target should work."""
        stacks = [1000, 2000, 3000]
        payouts = [0.5, 0.3, 0.2]

        eq_full = icm.equity(stacks, payouts)
        eq_sub = icm.equity_subset(stacks, payouts, targets=[1])

        np.testing.assert_allclose(eq_sub[1], eq_full[1], rtol=1e-10)

    def test_subset_non_target_zeros(self):
        """Non-target entries should be zero."""
        stacks = [1000, 2000, 3000, 4000]
        payouts = [0.5, 0.3, 0.2]
        targets = [1, 3]

        eq_sub = icm.equity_subset(stacks, payouts, targets)
        assert eq_sub[0] == 0.0
        assert eq_sub[2] == 0.0


class TestNumpyInput:
    """Test that numpy array inputs work correctly."""

    def test_numpy_stacks(self):
        """Numpy arrays for stacks should work."""
        stacks = np.array([1000.0, 2000.0, 3000.0])
        payouts = [0.5, 0.3, 0.2]
        eq = icm.equity(stacks, payouts)
        np.testing.assert_allclose(eq.sum(), sum(payouts), rtol=1e-8)

    def test_numpy_payouts(self):
        """Numpy arrays for payouts should work."""
        stacks = [1000, 2000, 3000]
        payouts = np.array([0.5, 0.3, 0.2])
        eq = icm.equity(stacks, payouts)
        np.testing.assert_allclose(eq.sum(), sum(payouts), rtol=1e-8)

    def test_numpy_both(self):
        """Numpy arrays for both stacks and payouts should work."""
        stacks = np.array([1000.0, 2000.0, 3000.0])
        payouts = np.array([0.5, 0.3, 0.2])
        eq = icm.equity(stacks, payouts)
        np.testing.assert_allclose(eq.sum(), sum(payouts), rtol=1e-8)

    def test_numpy_int_stacks(self):
        """Integer numpy arrays should be auto-converted to float64."""
        stacks = np.array([1000, 2000, 3000], dtype=np.int64)
        payouts = [0.5, 0.3, 0.2]
        eq = icm.equity(stacks, payouts)
        np.testing.assert_allclose(eq.sum(), sum(payouts), rtol=1e-8)

    def test_numpy_targets(self):
        """Numpy array for targets should work."""
        stacks = [1000, 2000, 3000]
        payouts = [0.5, 0.3, 0.2]
        targets = np.array([0, 2], dtype=np.int32)
        eq = icm.equity_subset(stacks, payouts, targets)
        assert eq[1] == 0.0  # non-target


class TestInputValidation:
    """Test error handling for invalid inputs."""

    def test_empty_stacks(self):
        with pytest.raises(ValueError, match="must not be empty"):
            icm.equity([], [0.5, 0.3, 0.2])

    def test_empty_payouts(self):
        with pytest.raises(ValueError, match="must not be empty"):
            icm.equity([1000, 2000], [])

    def test_k_greater_than_n(self):
        with pytest.raises(ValueError, match="k=3 must satisfy 1 <= k <= n=2"):
            icm.equity([1000, 2000], [0.5, 0.3, 0.2])

    def test_invalid_target_index(self):
        with pytest.raises(ValueError, match="target indices"):
            icm.equity_subset([1000, 2000, 3000], [0.5, 0.3], targets=[5])

    def test_negative_target_index(self):
        with pytest.raises(ValueError, match="target indices"):
            icm.equity_subset([1000, 2000, 3000], [0.5, 0.3], targets=[-1])

    def test_invalid_Q(self):
        with pytest.raises(ValueError, match="Q must be positive"):
            icm.equity([1000, 2000, 3000], [0.5, 0.3, 0.2], Q=0)
