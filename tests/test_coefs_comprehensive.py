# test_coefs_comprehensive.py

"""Comprehensive tests for coefficient computation."""

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdx.coefs import (
    coefficients,
    coefficients_non_uni,
    compute_coeffs,
    compute_inverse_Vandermonde,
)


class TestComputeInverseVandermonde:
    """Tests for inverse Vandermonde matrix computation."""

    def test_inverse_vandermonde_basic(self):
        """Test basic inverse Vandermonde computation."""
        c = compute_inverse_Vandermonde(column=1, offsets=[-1, 0, 1])
        # For first derivative with symmetric stencil
        expected = jnp.array([-0.5, 0, 0.5])
        assert jnp.allclose(c, expected)

    def test_inverse_vandermonde_second_derivative(self):
        """Test inverse Vandermonde for second derivative."""
        c = compute_inverse_Vandermonde(column=2, offsets=[-1, 0, 1])
        expected = jnp.array([1.0, -2.0, 1.0])
        assert jnp.allclose(c, expected)

    @pytest.mark.parametrize("offsets", [[-2, -1, 0, 1, 2], [-3, -2, -1, 0, 1, 2, 3]])
    def test_inverse_vandermonde_various_stencils(self, offsets):
        """Test inverse Vandermonde with various stencil sizes."""
        c = compute_inverse_Vandermonde(column=1, offsets=offsets)
        assert len(c) == len(offsets)
        assert not jnp.any(jnp.isnan(c))


class TestComputeCoeffs:
    """Tests for coefficient computation."""

    @pytest.mark.parametrize("analytic_inv", [True, False])
    def test_compute_coeffs_first_derivative(self, analytic_inv):
        """Test coefficient computation for first derivative."""
        coefs = compute_coeffs(1, [-1, 0, 1], analytic_inv=analytic_inv)

        assert jnp.allclose(coefs["coefficients"], jnp.array([-0.5, 0, 0.5]))
        assert jnp.allclose(coefs["offsets"], jnp.array([-1, 0, 1]))

    @pytest.mark.parametrize("analytic_inv", [True, False])
    def test_compute_coeffs_second_derivative(self, analytic_inv):
        """Test coefficient computation for second derivative."""
        coefs = compute_coeffs(2, [-1, 0, 1], analytic_inv=analytic_inv)

        assert jnp.allclose(coefs["coefficients"], jnp.array([1.0, -2.0, 1.0]))
        assert jnp.allclose(coefs["offsets"], jnp.array([-1, 0, 1]))

    @pytest.mark.parametrize("analytic_inv", [True, False])
    def test_compute_coeffs_asymmetric(self, analytic_inv):
        """Test coefficient computation for asymmetric stencil."""
        coefs = compute_coeffs(1, [-2, 0, 1], analytic_inv=analytic_inv)

        expected = jnp.array([-1.0 / 6, -0.5, 2.0 / 3])
        assert jnp.allclose(coefs["coefficients"], expected)


class TestCoefficientsUniform:
    """Tests for coefficients function (uniform grids)."""

    @pytest.mark.parametrize("analytic_inv", [True, False])
    @pytest.mark.parametrize("acc", [2, 4, 6, 8])
    def test_coefficients_first_derivative(self, acc, analytic_inv):
        """Test coefficients for first derivative with various accuracies."""
        c = coefficients(deriv=1, acc=acc, analytic_inv=analytic_inv)

        assert "center" in c
        assert "forward" in c
        assert "backward" in c

        # Central coefficients should be symmetric
        center_coefs = c["center"]["coefficients"]
        assert jnp.allclose(center_coefs, -center_coefs[::-1], atol=1e-5)

    @pytest.mark.parametrize("analytic_inv", [True, False])
    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_coefficients_second_derivative(self, acc, analytic_inv):
        """Test coefficients for second derivative."""
        c = coefficients(deriv=2, acc=acc, analytic_inv=analytic_inv)

        # Central coefficients should be symmetric
        center_coefs = c["center"]["coefficients"]
        assert jnp.allclose(center_coefs, center_coefs[::-1])

    def test_coefficients_order2_acc2(self):
        """Test specific case: 2nd derivative, 2nd order accuracy."""
        c = coefficients(deriv=2, acc=2)

        # Standard central difference for 2nd derivative
        assert jnp.allclose(c["center"]["coefficients"], jnp.array([1.0, -2.0, 1.0]))
        assert jnp.allclose(c["center"]["offsets"], jnp.array([-1, 0, 1]))

    def test_coefficients_order1_acc2(self):
        """Test specific case: 1st derivative, 2nd order accuracy."""
        c = coefficients(deriv=1, acc=2)

        assert jnp.allclose(c["center"]["coefficients"], jnp.array([-0.5, 0, 0.5]))
        assert jnp.allclose(c["center"]["offsets"], jnp.array([-1, 0, 1]))

    def test_coefficients_order1_acc4(self):
        """Test specific case: 1st derivative, 4th order accuracy."""
        c = coefficients(deriv=1, acc=4)

        expected_center = jnp.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        assert jnp.allclose(c["center"]["coefficients"], expected_center, atol=1e-6)
        assert jnp.allclose(
            c["center"]["offsets"], jnp.array([-2, -1, 0, 1, 2]), atol=1e-6
        )


class TestCoefficientsNonUniform:
    """Tests for non-uniform grid coefficients."""

    def test_coefficients_non_uni_uniform_grid(self):
        """Test that non-uniform coefficients match uniform for uniform grid."""
        # Create uniform grid
        x = jnp.linspace(0, 10, 100)
        dx = x[1] - x[0]

        # Uniform coefficients
        c_uni = coefficients(deriv=2, acc=2)
        coefs_uni = c_uni["center"]["coefficients"] / dx**2

        # Non-uniform coefficients on uniform grid
        c_non_uni = coefficients_non_uni(deriv=2, acc=2, coords=x, idx=50)
        coefs_non_uni = c_non_uni["coefficients"]

        assert jnp.allclose(coefs_non_uni, coefs_uni, atol=1e-6)

    def test_coefficients_non_uni_forward(self):
        """Test non-uniform coefficients at boundary (forward difference)."""
        x = jnp.array([0, 0.1, 0.3, 0.6, 1.0])
        c = coefficients_non_uni(deriv=1, acc=2, coords=x, idx=0)

        # Should use forward difference
        assert len(c["coefficients"]) >= 3
        assert jnp.min(c["offsets"]) >= 0

    def test_coefficients_non_uni_backward(self):
        """Test non-uniform coefficients at boundary (backward difference)."""
        x = jnp.array([0, 0.1, 0.3, 0.6, 1.0])
        c = coefficients_non_uni(deriv=1, acc=2, coords=x, idx=len(x) - 1)

        # Should use backward difference
        assert len(c["coefficients"]) >= 3
        assert jnp.max(c["offsets"]) <= 0

    def test_coefficients_non_uni_central(self):
        """Test non-uniform coefficients in interior (central difference)."""
        x = jnp.array([0, 0.1, 0.3, 0.6, 1.0])
        c = coefficients_non_uni(deriv=1, acc=2, coords=x, idx=2)

        # Should use central difference
        assert len(c["coefficients"]) >= 3


class TestCoefficientValidation:
    """Tests for coefficient validation and error handling."""

    def test_invalid_accuracy_odd(self):
        """Test that odd accuracy raises error."""
        with pytest.raises(ValueError):
            coefficients(deriv=1, acc=3)

    def test_invalid_accuracy_zero(self):
        """Test that zero accuracy raises error."""
        with pytest.raises(ValueError):
            coefficients(deriv=1, acc=0)

    def test_invalid_accuracy_negative(self):
        """Test that negative accuracy raises error."""
        with pytest.raises(ValueError):
            coefficients(deriv=1, acc=-2)

    def test_invalid_derivative_negative(self):
        """Test that negative derivative order raises error."""
        with pytest.raises(ValueError):
            coefficients(deriv=-1, acc=2)

    def test_insufficient_offsets(self):
        """Test that insufficient offsets raises error."""
        with pytest.raises(ValueError):
            coefficients(deriv=2, offsets=[-1, 1])  # Need at least 3 points

    def test_both_acc_and_offsets(self):
        """Test that specifying both acc and offsets raises error."""
        with pytest.raises(ValueError):
            coefficients(deriv=1, acc=2, offsets=[-1, 0, 1])

    def test_neither_acc_nor_offsets(self):
        """Test that specifying neither acc nor offsets raises error."""
        with pytest.raises(ValueError):
            coefficients(deriv=1)


class TestCoefficientAccuracy:
    """Tests for accuracy calculation in coefficients."""

    @pytest.mark.parametrize("deriv", [1, 2, 3])
    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_accuracy_matches_request(self, deriv, acc):
        """Test that computed accuracy matches requested accuracy."""
        c = coefficients(deriv=deriv, acc=acc)
        # Accuracy should be at least what was requested
        assert c["center"]["accuracy"] >= acc

    def test_accuracy_calculation_central(self):
        """Test accuracy calculation for central difference."""
        coefs = compute_coeffs(2, [-1, 0, 1])
        assert coefs["accuracy"] == 2

    def test_accuracy_calculation_forward(self):
        """Test accuracy calculation for forward difference."""
        coefs = compute_coeffs(1, [-1, 0])
        assert coefs["accuracy"] == 1


class TestCoefficientPropertyBased:
    """Property-based tests for coefficients."""

    @settings(deadline=None, max_examples=30)
    @given(
        acc=st.sampled_from([2, 4, 6]),
        deriv=st.integers(min_value=1, max_value=3),
    )
    def test_coefficients_symmetry(self, acc, deriv):
        """Property: central difference coefficients have appropriate symmetry."""
        c = coefficients(deriv=deriv, acc=acc)
        center_coefs = c["center"]["coefficients"]

        if deriv % 2 == 0:
            # Even derivatives: symmetric
            assert jnp.allclose(center_coefs, center_coefs[::-1], atol=1e-4)
        else:
            # Odd derivatives: antisymmetric
            assert jnp.allclose(center_coefs, -center_coefs[::-1], atol=1e-4)

    @settings(deadline=None, max_examples=30)
    @given(acc=st.sampled_from([2, 4, 6]))
    def test_coefficients_sum_first_derivative(self, acc):
        """Property: coefficients for 1st derivative should sum to 0."""
        c = coefficients(deriv=1, acc=acc)

        # All schemes should sum to 0 for first derivative
        assert jnp.isclose(jnp.sum(c["center"]["coefficients"]), 0, atol=1e-6)
        assert jnp.isclose(jnp.sum(c["forward"]["coefficients"]), 0, atol=1e-6)
        assert jnp.isclose(jnp.sum(c["backward"]["coefficients"]), 0, atol=1e-6)
