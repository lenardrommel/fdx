"""Comprehensive tests for Divergence, Curl, and Laplacian operators."""

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdx import Curl, Divergence, Laplacian


def tolerance_for_accuracy(acc, deriv=1):
    """Calculate tolerance based on accuracy order and derivative order."""
    base_tol = 10 ** (-acc / 2)
    return base_tol * (10 ** (deriv - 1))



class TestDivergenceBasic:
    """Basic functionality tests for Divergence operator."""

    def test_divergence_creation_2d(self):
        """Test basic Divergence operator creation for 2D."""
        div = Divergence(h=[0.1, 0.1])
        assert div.ndims == 2

    def test_divergence_creation_3d(self):
        """Test basic Divergence operator creation for 3D."""
        div = Divergence(h=[0.1, 0.1, 0.1])
        assert div.ndims == 3


class TestDivergence2D:
    """Tests for 2D Divergence operator."""

    def test_divergence_2d_linear_field(self, small_grid_2d):
        """Test divergence of linear vector field: F = [x, y]."""
        X, Y, dx, dy = small_grid_2d
        # Vector field F = [x, y]
        F = jnp.stack([X, Y], axis=0)

        div = Divergence(h=[dx, dy], acc=4)
        result = div(F)

        # div F = ∂x/∂x + ∂y/∂y = 1 + 1 = 2
        expected = 2 * jnp.ones_like(X)

        assert jnp.allclose(result, expected, atol=1e-10)

    def test_divergence_2d_quadratic_field(self, small_grid_2d):
        """Test divergence of quadratic field: F = [x^2, y^2]."""
        X, Y, dx, dy = small_grid_2d
        F = jnp.stack([X**2, Y**2], axis=0)

        div = Divergence(h=[dx, dy], acc=4)
        result = div(F)

        # div F = 2x + 2y
        expected = 2 * X + 2 * Y

        tol = tolerance_for_accuracy(4)
        assert jnp.allclose(result, expected, atol=tol)

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_divergence_2d_polynomial(self, medium_grid_2d, acc):
        """Test divergence of polynomial field: F = [x^2 + y, x + y^2]."""
        X, Y, dx, dy = medium_grid_2d
        F = jnp.stack([X**2 + Y, X + Y**2], axis=0)

        div = Divergence(h=[dx, dy], acc=acc)
        result = div(F)

        # div F = 2x + 2y
        expected = 2 * X + 2 * Y

        tol = tolerance_for_accuracy(acc)
        assert jnp.allclose(result, expected, atol=tol)


class TestDivergence3D:
    """Tests for 3D Divergence operator."""

    def test_divergence_3d_linear_field(self, small_grid_3d):
        """Test divergence of linear 3D field: F = [x, y, z]."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        F = jnp.stack([X, Y, Z], axis=0)

        div = Divergence(h=[dx, dy, dz], acc=4)
        result = div(F)

        # div F = 1 + 1 + 1 = 3
        expected = 3 * jnp.ones_like(X)

        assert jnp.allclose(result, expected, atol=1e-10)

    def test_divergence_3d_quadratic_field(self, small_grid_3d):
        """Test divergence of quadratic 3D field: F = [x^2, y^2, z^2]."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        F = jnp.stack([X**2, Y**2, Z**2], axis=0)

        div = Divergence(h=[dx, dy, dz], acc=4)
        result = div(F)

        # div F = 2x + 2y + 2z
        expected = 2 * (X + Y + Z)

        tol = tolerance_for_accuracy(4)
        assert jnp.allclose(result, expected, atol=tol)

    def test_divergence_3d_trigonometric(self, small_grid_3d):
        """Test divergence of trigonometric field."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        # F = [sin(x), cos(y), sin(z)]
        F = jnp.stack([jnp.sin(X), jnp.cos(Y), jnp.sin(Z)], axis=0)

        div = Divergence(h=[dx, dy, dz], acc=6)
        result = div(F)

        # div F = cos(x) - sin(y) + cos(z)
        expected = jnp.cos(X) - jnp.sin(Y) + jnp.cos(Z)

        tol = tolerance_for_accuracy(6)
        assert jnp.allclose(result, expected, atol=tol)


class TestCurlBasic:
    """Basic functionality tests for Curl operator."""

    def test_curl_creation_3d(self):
        """Test basic Curl operator creation."""
        curl = Curl(h=[0.1, 0.1, 0.1])
        assert curl.ndims == 3

    def test_curl_dimension_validation(self):
        """Test that Curl only works in 3D."""
        with pytest.raises(ValueError):
            Curl(h=[0.1, 0.1])  # 2D should fail


class TestCurl3D:
    """Tests for 3D Curl operator."""

    def test_curl_gradient_field(self, small_grid_3d):
        """Test that curl of gradient field is zero."""
        X, Y, Z, dx, dy, dz = small_grid_3d

        # Gradient field of scalar function f = x^2 + y^2 + z^2
        # grad f = [2x, 2y, 2z]
        F = jnp.stack([2 * X, 2 * Y, 2 * Z], axis=0)

        curl = Curl(h=[dx, dy, dz], acc=4)
        result = curl(F)

        # Curl of gradient should be zero
        expected = jnp.zeros_like(F)

        assert jnp.allclose(result, expected, atol=1e-8)

    def test_curl_simple_field(self, small_grid_3d):
        """Test curl of simple vector field: F = [-y, x, 0]."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        F = jnp.stack([-Y, X, jnp.zeros_like(Z)], axis=0)

        curl = Curl(h=[dx, dy, dz], acc=4)
        result = curl(F)

        # curl F = [0, 0, 2]
        expected_x = jnp.zeros_like(X)
        expected_y = jnp.zeros_like(Y)
        expected_z = 2 * jnp.ones_like(Z)

        tol = tolerance_for_accuracy(4)
        assert jnp.allclose(result[0], expected_x, atol=tol)
        assert jnp.allclose(result[1], expected_y, atol=tol)
        assert jnp.allclose(result[2], expected_z, atol=tol)

    def test_curl_polynomial_field(self, small_grid_3d):
        """Test curl of polynomial field: F = [y*z, x*z, x*y]."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        F = jnp.stack([Y * Z, X * Z, X * Y], axis=0)

        curl = Curl(h=[dx, dy, dz], acc=6)
        result = curl(F)

        # curl F = [x - x, y - y, z - z] = [0, 0, 0]
        expected = jnp.zeros_like(F)

        tol = tolerance_for_accuracy(6)
        assert jnp.allclose(result, expected, atol=tol)


class TestLaplacianBasic:
    """Basic functionality tests for Laplacian operator."""

    def test_laplacian_creation_1d(self):
        """Test basic Laplacian operator creation for 1D."""
        lap = Laplacian(h=[0.1])
        assert len(lap.h) == 1

    def test_laplacian_creation_2d(self):
        """Test basic Laplacian operator creation for 2D."""
        lap = Laplacian(h=[0.1, 0.1])
        assert len(lap.h) == 2

    def test_laplacian_creation_3d(self):
        """Test basic Laplacian operator creation for 3D."""
        lap = Laplacian(h=[0.1, 0.1, 0.1])
        assert len(lap.h) == 3


class TestLaplacian1D:
    """Tests for 1D Laplacian operator."""

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_laplacian_1d_quadratic(self, small_grid_1d, acc):
        """Test Laplacian of quadratic: f(x) = x^2."""
        x, dx = small_grid_1d
        f = x**2

        lap = Laplacian(h=[dx], acc=acc)
        result = lap(f)

        # Laplacian of x^2 is 2
        expected = 2 * jnp.ones_like(x)

        tol = tolerance_for_accuracy(acc, deriv=2)
        assert jnp.allclose(result, expected, atol=tol)

    def test_laplacian_1d_quartic(self, medium_grid_1d):
        """Test Laplacian of quartic: f(x) = x^4."""
        x, dx = medium_grid_1d
        f = x**4

        lap = Laplacian(h=[dx], acc=4)
        result = lap(f)

        # Laplacian of x^4 is 12x^2
        expected = 12 * x**2

        tol = tolerance_for_accuracy(4, deriv=2)
        assert jnp.allclose(result, expected, atol=tol)

    def test_laplacian_1d_sine(self, medium_grid_1d):
        """Test Laplacian of sine: f(x) = sin(x)."""
        x, dx = medium_grid_1d
        f = jnp.sin(x)

        lap = Laplacian(h=[dx], acc=6)
        result = lap(f)

        # Laplacian of sin(x) is -sin(x)
        expected = -jnp.sin(x)

        tol = tolerance_for_accuracy(6, deriv=2)
        assert jnp.allclose(result, expected, atol=tol)


class TestLaplacian2D:
    """Tests for 2D Laplacian operator."""

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_laplacian_2d_quadratic(self, small_grid_2d, acc):
        """Test Laplacian of f(x,y) = x^2 + y^2."""
        X, Y, dx, dy = small_grid_2d
        f = X**2 + Y**2

        lap = Laplacian(h=[dx, dy], acc=acc)
        result = lap(f)

        # ∇²(x^2 + y^2) = 2 + 2 = 4
        expected = 4 * jnp.ones_like(X)

        tol = tolerance_for_accuracy(acc, deriv=2)
        assert jnp.allclose(result, expected, atol=tol)

    def test_laplacian_2d_polynomial(self, medium_grid_2d):
        """Test Laplacian of f(x,y) = x^3 + y^3."""
        X, Y, dx, dy = medium_grid_2d
        f = X**3 + Y**3

        lap = Laplacian(h=[dx, dy], acc=4)
        result = lap(f)

        # ∇²(x^3 + y^3) = 6x + 6y
        expected = 6 * X + 6 * Y

        tol = tolerance_for_accuracy(4, deriv=2)
        assert jnp.allclose(result, expected, atol=tol)

    def test_laplacian_2d_trigonometric(self, medium_grid_2d):
        """Test Laplacian of f(x,y) = sin(x) * sin(y)."""
        X, Y, dx, dy = medium_grid_2d
        f = jnp.sin(X) * jnp.sin(Y)

        lap = Laplacian(h=[dx, dy], acc=6)
        result = lap(f)

        # ∇²(sin(x)sin(y)) = -2sin(x)sin(y)
        expected = -2 * jnp.sin(X) * jnp.sin(Y)

        tol = tolerance_for_accuracy(6, deriv=2)
        assert jnp.allclose(result, expected, atol=tol)


class TestLaplacian3D:
    """Tests for 3D Laplacian operator."""

    def test_laplacian_3d_quadratic(self, small_grid_3d):
        """Test Laplacian of f(x,y,z) = x^2 + y^2 + z^2."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        f = X**2 + Y**2 + Z**2

        lap = Laplacian(h=[dx, dy, dz], acc=4)
        result = lap(f)

        # ∇²(x^2 + y^2 + z^2) = 2 + 2 + 2 = 6
        expected = 6 * jnp.ones_like(X)

        tol = tolerance_for_accuracy(4, deriv=2)
        assert jnp.allclose(result, expected, atol=tol)

    def test_laplacian_3d_trigonometric(self, small_grid_3d):
        """Test Laplacian of f(x,y,z) = sin(x)*sin(y)*sin(z)."""
        X, Y, Z, dx, dy, dz = small_grid_3d
        f = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)

        lap = Laplacian(h=[dx, dy, dz], acc=6)
        result = lap(f)

        # ∇²(sin(x)sin(y)sin(z)) = -3sin(x)sin(y)sin(z)
        expected = -3 * jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)

        tol = tolerance_for_accuracy(6, deriv=2)
        assert jnp.allclose(result, expected, atol=tol)


class TestVectorOperatorsPropertyBased:
    """Property-based tests for vector operators."""

    @settings(deadline=None, max_examples=30)
    @given(
        a=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
        b=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
    )
    def test_divergence_linear_field_property(self, a, b):
        """Property: div([ax, by]) = a + b."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        F = jnp.stack([a * X, b * Y], axis=0)
        div = Divergence(h=[dx, dy], acc=6)
        result = div(F)

        expected = (a + b) * jnp.ones_like(X)

        assert jnp.allclose(result, expected, rtol=1e-4, atol=1e-8)

    @settings(deadline=None, max_examples=30)
    @given(
        a=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False)
    )
    def test_laplacian_quadratic_property(self, a):
        """Property: Laplacian of ax^2 + ay^2 is 4a."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = a * (X**2 + Y**2)
        lap = Laplacian(h=[dx, dy], acc=6)
        result = lap(f)

        expected = 4 * a * jnp.ones_like(X)

        assert jnp.allclose(result, expected, rtol=1e-3, atol=1e-6)
