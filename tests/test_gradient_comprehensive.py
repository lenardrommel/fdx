"""Comprehensive tests for the Gradient operator."""

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdx import Gradient


def tolerance_for_accuracy(acc, deriv=1):
    """Calculate tolerance based on accuracy order and derivative order."""
    base_tol = 10 ** (-acc / 2)
    return base_tol * (10 ** (deriv - 1))



class TestGradientBasic:
    """Basic functionality tests for Gradient operator."""

    def test_gradient_creation_1d(self):
        """Test basic Gradient operator creation for 1D."""
        grad = Gradient(h=[0.1])
        assert grad.ndims == 1
        assert grad.acc == 2

    def test_gradient_creation_2d(self):
        """Test basic Gradient operator creation for 2D."""
        grad = Gradient(h=[0.1, 0.1])
        assert grad.ndims == 2
        assert grad.acc == 2

    def test_gradient_creation_3d(self):
        """Test basic Gradient operator creation for 3D."""
        grad = Gradient(h=[0.1, 0.1, 0.1])
        assert grad.ndims == 3
        assert grad.acc == 2

    def test_gradient_with_custom_accuracy(self):
        """Test Gradient with custom accuracy."""
        grad = Gradient(h=[0.1, 0.1], acc=6)
        assert grad.acc == 6


class TestGradient1D:
    """Tests for 1D Gradient operator."""

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_gradient_1d_linear(self, small_grid_1d, acc):
        """Test gradient of linear function: f(x) = 3x + 2."""
        x, dx = small_grid_1d
        f = 3 * x + 2
        grad = Gradient(h=[dx], acc=acc)

        actual = grad(f, axis=0)
        expected = 3 * jnp.ones_like(x)

        # Linear should be exact
        assert jnp.allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_gradient_1d_quadratic(self, small_grid_1d, acc):
        """Test gradient of quadratic: f(x) = x^2."""
        x, dx = small_grid_1d
        f = x**2
        grad = Gradient(h=[dx], acc=acc)

        actual = grad(f, axis=0)
        expected = 2 * x

        tol = tolerance_for_accuracy(acc)
        assert jnp.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("acc", [2, 4, 6, 8])
    def test_gradient_1d_sine(self, medium_grid_1d, acc):
        """Test gradient of sine function."""
        x, dx = medium_grid_1d
        f = jnp.sin(x)
        grad = Gradient(h=[dx], acc=acc)

        actual = grad(f, axis=0)
        expected = jnp.cos(x)

        tol = tolerance_for_accuracy(acc)
        assert jnp.allclose(actual, expected, atol=tol)

    def test_gradient_1d_polynomial(self, medium_grid_1d):
        """Test gradient of polynomial: f(x) = x^3 + 2x^2 + x."""
        x, dx = medium_grid_1d
        f = x**3 + 2 * x**2 + x
        grad = Gradient(h=[dx], acc=4)

        actual = grad(f, axis=0)
        expected = 3 * x**2 + 4 * x + 1

        tol = tolerance_for_accuracy(4)
        assert jnp.allclose(actual, expected, atol=tol)


class TestGradient2D:
    """Tests for 2D Gradient operator."""

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_gradient_2d_quadratic(self, small_grid_2d, acc):
        """Test gradient of 2D quadratic: f(x,y) = x^2 + y^2."""
        X, Y, dx, dy = small_grid_2d
        f = X**2 + Y**2
        grad = Gradient(h=[dx, dy], acc=acc)

        grad_f = grad(f)
        expected_x = 2 * X
        expected_y = 2 * Y

        tol = tolerance_for_accuracy(acc)
        assert jnp.allclose(grad_f[0], expected_x, atol=tol)
        assert jnp.allclose(grad_f[1], expected_y, atol=tol)

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_gradient_2d_polynomial(self, medium_grid_2d, acc):
        """Test gradient of 2D polynomial: f(x,y) = x^3 + 2xy + y^2."""
        X, Y, dx, dy = medium_grid_2d
        f = X**3 + 2 * X * Y + Y**2
        grad = Gradient(h=[dx, dy], acc=acc)

        grad_f = grad(f)
        expected_x = 3 * X**2 + 2 * Y
        expected_y = 2 * X + 2 * Y

        tol = tolerance_for_accuracy(acc)
        assert jnp.allclose(grad_f[0], expected_x, atol=tol)
        assert jnp.allclose(grad_f[1], expected_y, atol=tol)

    def test_gradient_2d_trigonometric(self, medium_grid_2d):
        """Test gradient of f(x,y) = sin(x) * cos(y)."""
        X, Y, dx, dy = medium_grid_2d
        f = jnp.sin(X) * jnp.cos(Y)
        grad = Gradient(h=[dx, dy], acc=6)

        grad_f = grad(f)
        expected_x = jnp.cos(X) * jnp.cos(Y)
        expected_y = -jnp.sin(X) * jnp.sin(Y)

        tol = tolerance_for_accuracy(6)
        assert jnp.allclose(grad_f[0], expected_x, atol=tol)
        assert jnp.allclose(grad_f[1], expected_y, atol=tol)

    def test_gradient_2d_single_axis(self, small_grid_2d):
        """Test gradient computation along a single axis."""
        X, Y, dx, dy = small_grid_2d
        f = X**2 + Y**2
        grad = Gradient(h=[dx, dy], acc=4)

        # Compute gradient along x-axis only
        grad_x = grad(f, axis=0)
        expected_x = 2 * X

        assert jnp.allclose(grad_x, expected_x, atol=1e-6)


class TestGradient3D:
    """Tests for 3D Gradient operator."""

    def test_gradient_3d_quadratic(self, small_grid_3d):
        """Test gradient of 3D quadratic: f(x,y,z) = x^2 + y^2 + z^2."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        f = X**2 + Y**2 + Z**2
        grad = Gradient(h=[dx, dy, dz], acc=4)

        grad_f = grad(f)
        expected_x = 2 * X
        expected_y = 2 * Y
        expected_z = 2 * Z

        assert jnp.allclose(grad_f[0], expected_x, atol=1e-6)
        assert jnp.allclose(grad_f[1], expected_y, atol=1e-6)
        assert jnp.allclose(grad_f[2], expected_z, atol=1e-6)

    def test_gradient_3d_trigonometric(self, small_grid_3d):
        """Test gradient of f(x,y,z) = sin(x) * sin(y) * sin(z)."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        f = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
        grad = Gradient(h=[dx, dy, dz], acc=4)

        grad_f = grad(f)
        expected_x = jnp.cos(X) * jnp.sin(Y) * jnp.sin(Z)
        expected_y = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)
        expected_z = jnp.sin(X) * jnp.sin(Y) * jnp.cos(Z)

        tol = tolerance_for_accuracy(4)
        assert jnp.allclose(grad_f[0], expected_x, atol=tol)
        assert jnp.allclose(grad_f[1], expected_y, atol=tol)
        assert jnp.allclose(grad_f[2], expected_z, atol=tol)


class TestGradientEdgeCases:
    """Test edge cases and special scenarios."""

    def test_gradient_constant_function(self, small_grid_2d):
        """Test gradient of constant function."""
        X, Y, dx, dy = small_grid_2d
        f = jnp.ones_like(X) * 5.0
        grad = Gradient(h=[dx, dy], acc=4)

        grad_f = grad(f)

        # Gradient of constant should be zero
        assert jnp.allclose(grad_f[0], 0, atol=1e-10)
        assert jnp.allclose(grad_f[1], 0, atol=1e-10)

    def test_gradient_dimension_validation(self, small_grid_2d):
        """Test that gradient validates dimensions correctly."""
        X, Y, dx, dy = small_grid_2d
        f = X**2 + Y**2
        grad_2d = Gradient(h=[dx, dy], acc=4)

        result = grad_2d(f)
        assert result.shape == (2, *f.shape)

    def test_gradient_fine_vs_coarse_grid(self):
        """Test gradient on fine vs coarse grids."""
        # Fine grid
        x_fine = jnp.linspace(0, 2 * jnp.pi, 200)
        dx_fine = x_fine[1] - x_fine[0]
        f_fine = jnp.sin(x_fine)
        grad_fine = Gradient(h=[dx_fine], acc=4)
        result_fine = grad_fine(f_fine, axis=0)
        error_fine = jnp.max(jnp.abs(result_fine - jnp.cos(x_fine)))

        # Coarse grid
        x_coarse = jnp.linspace(0, 2 * jnp.pi, 50)
        dx_coarse = x_coarse[1] - x_coarse[0]
        f_coarse = jnp.sin(x_coarse)
        grad_coarse = Gradient(h=[dx_coarse], acc=4)
        result_coarse = grad_coarse(f_coarse, axis=0)
        error_coarse = jnp.max(jnp.abs(result_coarse - jnp.cos(x_coarse)))

        # Finer grid should have smaller error
        assert error_fine < error_coarse


class TestGradientAccuracy:
    """Test accuracy convergence properties."""

    def test_accuracy_order_convergence(self):
        """Test that error decreases with higher accuracy order."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]
        f = jnp.sin(2 * x)
        expected = 2 * jnp.cos(2 * x)

        errors = []
        for acc in [2, 4, 6, 8]:
            grad = Gradient(h=[dx], acc=acc)
            result = grad(f, axis=0)
            error = jnp.max(jnp.abs(result - expected))
            errors.append(error)

        # Each increase in accuracy should reduce error
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

    def test_grid_refinement_convergence(self):
        """Test that error decreases with grid refinement."""
        errors = []
        grid_sizes = [25, 50, 100, 200]

        for n in grid_sizes:
            x = jnp.linspace(0, 2 * jnp.pi, n)
            dx = x[1] - x[0]
            f = jnp.sin(x)
            grad = Gradient(h=[dx], acc=4)
            result = grad(f, axis=0)
            error = jnp.max(jnp.abs(result - jnp.cos(x)))
            errors.append(error)

        # Finer grids should have smaller errors
        assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))


class TestGradientPropertyBased:
    """Property-based tests using Hypothesis."""

    @settings(deadline=None, max_examples=30)
    @given(
        a=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
        b=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
    )
    def test_gradient_2d_quadratic_property(self, a, b):
        """Property: gradient of f(x,y) = ax^2 + by^2 is [2ax, 2by]."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = a * X**2 + b * Y**2
        grad = Gradient(h=[dx, dy], acc=6)
        grad_f = grad(f)

        expected_x = 2 * a * X
        expected_y = 2 * b * Y

        assert jnp.allclose(grad_f[0], expected_x, rtol=1e-4, atol=1e-6)
        assert jnp.allclose(grad_f[1], expected_y, rtol=1e-4, atol=1e-6)

    @settings(deadline=None, max_examples=30)
    @given(
        freq=st.floats(min_value=0.5, max_value=3, allow_nan=False, allow_infinity=False)
    )
    def test_gradient_sine_frequency_property(self, freq):
        """Property: gradient of sin(freq*x) is freq*cos(freq*x)."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]
        f = jnp.sin(freq * x)

        grad = Gradient(h=[dx], acc=6)
        result = grad(f, axis=0)
        expected = freq * jnp.cos(freq * x)

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-5)
