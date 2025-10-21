"""Comprehensive tests for the Diff operator."""

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdx import Diff
from fdx.grids import EquidistantAxis


def tolerance_for_accuracy(acc, deriv=1):
    """Calculate tolerance based on accuracy order and derivative order."""
    base_tol = 10 ** (-acc / 2)
    return base_tol * (10 ** (deriv - 1))


class TestDiffBasic:
    """Basic functionality tests for Diff operator."""

    def test_diff_creation(self):
        """Test basic Diff operator creation."""
        d_dx = Diff(0, EquidistantAxis(0, 0.1))
        assert d_dx.dim == 0
        assert d_dx.order == 1
        assert d_dx.acc == 2

    def test_diff_with_custom_accuracy(self):
        """Test Diff operator with custom accuracy."""
        d_dx = Diff(0, EquidistantAxis(0, 0.1), acc=4)
        assert d_dx.acc == 4

    def test_diff_dimension_validation(self, small_grid_1d):
        """Test that Diff validates input dimensions."""
        x, dx = small_grid_1d
        f = x**2
        d_dx = Diff(0, EquidistantAxis(0, dx))
        result = d_dx(f)
        assert result.shape == f.shape


class TestDiffPolynomials:
    """Test Diff operator on polynomial functions."""

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_first_derivative_linear(self, small_grid_1d, acc):
        """Test first derivative of linear function: f(x) = 2x + 1."""
        x, dx = small_grid_1d
        f = 2 * x + 1
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=acc)

        actual = d_dx(f)
        expected = 2 * jnp.ones_like(x)

        # Linear function should be exact for any accuracy
        assert jnp.allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_first_derivative_quadratic(self, small_grid_1d, acc):
        """Test first derivative of quadratic: f(x) = x^2."""
        x, dx = small_grid_1d
        f = x**2
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=acc)

        actual = d_dx(f)
        expected = 2 * x

        tol = tolerance_for_accuracy(acc, deriv=1)
        assert jnp.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_first_derivative_cubic(self, medium_grid_1d, acc):
        """Test first derivative of cubic: f(x) = x^3."""
        x, dx = medium_grid_1d
        f = x**3
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=acc)

        actual = d_dx(f)
        expected = 3 * x**2

        tol = tolerance_for_accuracy(acc, deriv=1)
        assert jnp.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_second_derivative_quadratic(self, small_grid_1d, acc):
        """Test second derivative of quadratic: f(x) = x^2."""
        x, dx = small_grid_1d
        f = x**2
        d2_dx2 = Diff(0, EquidistantAxis(0, dx), acc=acc) ** 2

        actual = d2_dx2(f)
        expected = 2 * jnp.ones_like(x)

        tol = tolerance_for_accuracy(acc, deriv=2)
        assert jnp.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_second_derivative_quartic(self, medium_grid_1d, acc):
        """Test second derivative of quartic: f(x) = x^4."""
        x, dx = medium_grid_1d
        f = x**4
        d2_dx2 = Diff(0, EquidistantAxis(0, dx), acc=acc) ** 2

        actual = d2_dx2(f)
        expected = 12 * x**2

        tol = tolerance_for_accuracy(acc, deriv=2)
        assert jnp.allclose(actual, expected, atol=tol)


class TestDiffTrigonometric:
    """Test Diff operator on trigonometric functions."""

    @pytest.mark.parametrize("acc", [2, 4, 6, 8])
    def test_derivative_sine(self, medium_grid_1d, acc):
        """Test derivative of sin(x)."""
        x, dx = medium_grid_1d
        f = jnp.sin(x)
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=acc)

        actual = d_dx(f)
        expected = jnp.cos(x)

        tol = tolerance_for_accuracy(acc, deriv=1)
        assert jnp.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("acc", [2, 4, 6, 8])
    def test_second_derivative_sine(self, medium_grid_1d, acc):
        """Test second derivative of sin(x)."""
        x, dx = medium_grid_1d
        f = jnp.sin(x)
        d2_dx2 = Diff(0, EquidistantAxis(0, dx), acc=acc) ** 2

        actual = d2_dx2(f)
        expected = -jnp.sin(x)

        tol = tolerance_for_accuracy(acc, deriv=2)
        assert jnp.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("freq", [1.0, 2.0, 3.0])
    def test_derivative_sine_frequency(self, medium_grid_1d, freq):
        """Test derivative of sin(freq * x)."""
        x, dx = medium_grid_1d
        f = jnp.sin(freq * x)
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=4)

        actual = d_dx(f)
        expected = freq * jnp.cos(freq * x)

        tol = tolerance_for_accuracy(4, deriv=1)
        assert jnp.allclose(actual, expected, atol=tol)


class TestDiffMultidimensional:
    """Test Diff operator on multidimensional arrays."""

    def test_diff_2d_x_direction(self, small_grid_2d):
        """Test derivative in x direction on 2D grid."""
        X, Y, dx, dy = small_grid_2d
        f = X**2 + Y**2
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=4)

        actual = d_dx(f)
        expected = 2 * X

        assert jnp.allclose(actual, expected, atol=1e-6)

    def test_diff_2d_y_direction(self, small_grid_2d):
        """Test derivative in y direction on 2D grid."""
        X, Y, dx, dy = small_grid_2d
        f = X**2 + Y**2
        d_dy = Diff(1, EquidistantAxis(1, dy), acc=4)

        actual = d_dy(f)
        expected = 2 * Y

        assert jnp.allclose(actual, expected, atol=1e-6)

    def test_diff_2d_mixed_function(self, medium_grid_2d):
        """Test derivative on mixed 2D function."""
        X, Y, dx, dy = medium_grid_2d
        f = jnp.sin(X) * jnp.cos(Y)
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=6)

        actual = d_dx(f)
        expected = jnp.cos(X) * jnp.cos(Y)

        tol = tolerance_for_accuracy(6, deriv=1)
        assert jnp.allclose(actual, expected, atol=tol)


class TestDiffOperatorComposition:
    """Test composition and chaining of Diff operators."""

    def test_power_operator(self, medium_grid_1d):
        """Test power operator for higher derivatives."""
        x, dx = medium_grid_1d
        f = x**4
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=4)
        d2_dx2 = d_dx**2

        assert d2_dx2.order == 2

        actual = d2_dx2(f)
        expected = 12 * x**2

        tol = tolerance_for_accuracy(4, deriv=2)
        assert jnp.allclose(actual, expected, atol=tol)

    def test_multiplication_same_axis(self, medium_grid_1d):
        """Test multiplication of operators on same axis."""
        x, dx = medium_grid_1d
        f = x**4
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=4)
        d2_dx2 = d_dx * d_dx

        assert d2_dx2.order == 2

        actual = d2_dx2(f)
        expected = 12 * x**2

        tol = tolerance_for_accuracy(4, deriv=2)
        assert jnp.allclose(actual, expected, atol=tol)


class TestDiffAccuracy:
    """Test accuracy convergence of Diff operator."""

    def test_accuracy_convergence(self):
        """Test that higher accuracy produces lower errors."""
        accuracies = [2, 4, 6, 8]
        errors = []

        for acc in accuracies:
            x = jnp.linspace(0, 2 * jnp.pi, 100)
            dx = x[1] - x[0]
            f = jnp.sin(2 * x)
            d_dx = Diff(0, EquidistantAxis(0, dx), acc=acc)

            actual = d_dx(f)
            expected = 2 * jnp.cos(2 * x)
            error = jnp.max(jnp.abs(actual - expected))
            errors.append(error)

        # Higher accuracy should generally produce smaller errors
        assert errors[1] < errors[0], "4th order should be better than 2nd order"
        assert errors[2] < errors[1], "6th order should be better than 4th order"


class TestDiffPropertyBased:
    """Property-based tests using Hypothesis."""

    @settings(deadline=None, max_examples=50)
    @given(
        coeff=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        power=st.integers(min_value=1, max_value=3),
    )
    def test_derivative_polynomial(self, coeff, power):
        """Test derivative of polynomial c*x^n."""
        x = jnp.linspace(0.1, 2, 50)
        dx = x[1] - x[0]

        f = coeff * x**power
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=6)

        actual = d_dx(f)
        expected = coeff * power * x ** (power - 1)

        # Use relative tolerance for varying coefficients
        assert jnp.allclose(actual, expected, rtol=1e-3, atol=1e-6)

    @settings(deadline=None, max_examples=50)
    @given(
        freq=st.floats(min_value=0.5, max_value=5, allow_nan=False, allow_infinity=False)
    )
    def test_derivative_sine_property(self, freq):
        """Property: derivative of sin(freq*x) is freq*cos(freq*x)."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]

        f = jnp.sin(freq * x)
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=6)

        actual = d_dx(f)
        expected = freq * jnp.cos(freq * x)

        assert jnp.allclose(actual, expected, rtol=1e-3, atol=1e-5)
