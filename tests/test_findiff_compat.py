"""Comprehensive tests for FinDiff compatibility layer."""

import jax.numpy as jnp
import pytest

from fdx import FinDiff


class TestFinDiffBasic:
    """Basic functionality tests for FinDiff."""

    def test_findiff_creation_simple(self):
        """Test basic FinDiff creation with axis and spacing."""
        d_dx = FinDiff(0, 0.1)
        assert d_dx is not None

    def test_findiff_creation_with_derivative_order(self):
        """Test FinDiff creation with derivative order."""
        d2_dx2 = FinDiff(0, 0.1, 2)
        assert d2_dx2 is not None

    def test_findiff_creation_with_accuracy(self):
        """Test FinDiff creation with custom accuracy."""
        d_dx = FinDiff(0, 0.1, acc=4)
        assert d_dx is not None


class TestFinDiff1D:
    """Tests for 1D FinDiff operations."""

    def test_findiff_1d_first_derivative(self):
        """Test 1D first derivative with FinDiff."""
        x = jnp.linspace(0, 1, 50)
        dx = x[1] - x[0]
        f = x**2

        d_dx = FinDiff(0, dx, 1)
        result = d_dx(f)

        expected = 2 * x
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_findiff_1d_second_derivative(self):
        """Test 1D second derivative with FinDiff."""
        x = jnp.linspace(0, 1, 50)
        dx = x[1] - x[0]
        f = x**3

        d2_dx2 = FinDiff(0, dx, 2)
        result = d2_dx2(f)

        expected = 6 * x
        assert jnp.allclose(result, expected, atol=1e-2)

    @pytest.mark.parametrize("acc", [2, 4, 6, 10])
    def test_findiff_1d_various_accuracies(self, acc):
        """Test FinDiff with various accuracy orders."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]
        f = jnp.sin(x)

        d_dx = FinDiff(0, dx, acc=acc)
        result = d_dx(f)

        expected = jnp.cos(x)
        # Higher accuracy should give better results
        tol = 10 ** (-acc / 2)
        assert jnp.allclose(result, expected, atol=tol)

    def test_findiff_1d_polynomial(self):
        """Test FinDiff on polynomial function."""
        x = jnp.linspace(-1, 1, 60)
        dx = x[1] - x[0]
        f = x**4 + 2 * x**3 + x**2

        d_dx = FinDiff(0, dx, 1, acc=6)
        result = d_dx(f)

        expected = 4 * x**3 + 6 * x**2 + 2 * x
        assert jnp.allclose(result, expected, atol=1e-4)


class TestFinDiff2D:
    """Tests for 2D FinDiff operations."""

    def test_findiff_2d_x_derivative(self):
        """Test 2D derivative in x-direction."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]

        f = X**2 + Y**2
        d_dx = FinDiff(0, dx)
        result = d_dx(f)

        expected = 2 * X
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_findiff_2d_y_derivative(self):
        """Test 2D derivative in y-direction."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dy = y[1] - y[0]

        f = X**2 + Y**2
        d_dy = FinDiff(1, dy)
        result = d_dy(f)

        expected = 2 * Y
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_findiff_2d_second_derivative(self):
        """Test 2D second derivative."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]

        f = X**3 + Y**2
        d2_dx2 = FinDiff(0, dx, 2)
        result = d2_dx2(f)

        expected = 6 * X
        assert jnp.allclose(result, expected, atol=1e-2)


class TestFinDiff3D:
    """Tests for 3D FinDiff operations."""

    def test_findiff_3d_derivatives(self):
        """Test 3D derivatives in each direction."""
        x = jnp.linspace(0, 1, 20)
        y = jnp.linspace(0, 1, 20)
        z = jnp.linspace(0, 1, 20)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        f = X**2 + Y**2 + Z**2

        # Test x-derivative
        d_dx = FinDiff(0, dx)
        result_x = d_dx(f)
        expected_x = 2 * X
        assert jnp.allclose(result_x, expected_x, atol=1e-3)

        # Test y-derivative
        d_dy = FinDiff(1, dy)
        result_y = d_dy(f)
        expected_y = 2 * Y
        assert jnp.allclose(result_y, expected_y, atol=1e-3)

        # Test z-derivative
        d_dz = FinDiff(2, dz)
        result_z = d_dz(f)
        expected_z = 2 * Z
        assert jnp.allclose(result_z, expected_z, atol=1e-3)


class TestFinDiffMixed:
    """Tests for mixed derivatives with FinDiff."""

    def test_findiff_mixed_derivative_2d(self):
        """Test mixed partial derivative in 2D."""
        x = jnp.linspace(0, 1, 40)
        y = jnp.linspace(0, 1, 40)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = X**2 * Y**2

        # Mixed derivative: d²f/dxdy
        d_dxdy = FinDiff((0, dx), (1, dy))
        result = d_dxdy(f)

        # d/dx(d/dy(x²y²)) = d/dx(2x²y) = 4xy
        expected = 4 * X * Y

        assert jnp.allclose(result, expected, atol=1e-2)

    def test_findiff_mixed_higher_order(self):
        """Test mixed derivative with higher order."""
        x = jnp.linspace(0, 1, 40)
        y = jnp.linspace(0, 1, 40)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = X**3 * Y**2

        # d³f/dx²dy
        d_dx2dy = FinDiff((0, dx, 2), (1, dy))
        result = d_dx2dy(f)

        # d²/dx²(d/dy(x³y²)) = d²/dx²(2x³y) = 12xy
        expected = 12 * X * Y

        assert jnp.allclose(result, expected, atol=1e-2)


class TestFinDiffTrigonometric:
    """Tests for FinDiff on trigonometric functions."""

    def test_findiff_sine_1d(self):
        """Test FinDiff on sine function."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]
        f = jnp.sin(x)

        d_dx = FinDiff(0, dx, acc=6)
        result = d_dx(f)

        expected = jnp.cos(x)
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_findiff_cosine_second_derivative(self):
        """Test second derivative of cosine."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]
        f = jnp.cos(x)

        d2_dx2 = FinDiff(0, dx, 2, acc=6)
        result = d2_dx2(f)

        expected = -jnp.cos(x)
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_findiff_2d_trigonometric(self):
        """Test FinDiff on 2D trigonometric function."""
        x = jnp.linspace(0, 2 * jnp.pi, 50)
        y = jnp.linspace(0, 2 * jnp.pi, 50)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        y[1] - y[0]

        f = jnp.sin(X) * jnp.cos(Y)

        d_dx = FinDiff(0, dx, acc=6)
        result = d_dx(f)

        expected = jnp.cos(X) * jnp.cos(Y)
        assert jnp.allclose(result, expected, atol=1e-4)


class TestFinDiffEdgeCases:
    """Test edge cases and special scenarios."""

    def test_findiff_constant_function(self):
        """Test derivative of constant function."""
        x = jnp.linspace(0, 1, 50)
        dx = x[1] - x[0]
        f = jnp.ones_like(x) * 5.0

        d_dx = FinDiff(0, dx)
        result = d_dx(f)

        # Derivative of constant should be zero
        assert jnp.allclose(result, 0, atol=1e-10)

    def test_findiff_linear_exact(self):
        """Test that linear function derivative is exact."""
        x = jnp.linspace(0, 1, 50)
        dx = x[1] - x[0]
        f = 3 * x + 2

        d_dx = FinDiff(0, dx)
        result = d_dx(f)

        expected = 3 * jnp.ones_like(x)
        # Should be machine precision for linear
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_findiff_high_order_polynomial(self):
        """Test high-order polynomial."""
        x = jnp.linspace(0, 1, 100)
        dx = x[1] - x[0]
        f = x**6

        d_dx = FinDiff(0, dx, 1, acc=8)
        result = d_dx(f)

        expected = 6 * x**5
        assert jnp.allclose(result, expected, atol=1e-3)


class TestFinDiffAccuracy:
    """Test accuracy convergence properties."""

    def test_accuracy_improvement(self):
        """Test that higher accuracy gives better results."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]
        f = jnp.sin(2 * x)
        expected = 2 * jnp.cos(2 * x)

        errors = []
        for acc in [2, 4, 6, 8]:
            d_dx = FinDiff(0, dx, acc=acc)
            result = d_dx(f)
            error = jnp.max(jnp.abs(result - expected))
            errors.append(error)

        # Errors should generally decrease
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

    def test_grid_refinement(self):
        """Test that finer grids give better results."""
        errors = []
        grid_sizes = [25, 50, 100, 200]

        for n in grid_sizes:
            x = jnp.linspace(0, 2 * jnp.pi, n)
            dx = x[1] - x[0]
            f = jnp.sin(x)

            d_dx = FinDiff(0, dx, acc=4)
            result = d_dx(f)
            error = jnp.max(jnp.abs(result - jnp.cos(x)))
            errors.append(error)

        # Finer grids should have smaller errors
        assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))
