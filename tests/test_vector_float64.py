"""Tests that vector.py operators preserve float64 precision."""

import jax.numpy as jnp
import pytest

from fdx import Gradient, Divergence, Curl, Laplacian, Jacobian


def _interior(width=3):
    return slice(width, -width)


def _interior_2d(width=3):
    return (slice(width, -width), slice(width, -width))


def _interior_3d(width=3):
    return (slice(width, -width), slice(width, -width), slice(width, -width))


class TestGradientFloat64:
    """Gradient operator with float64 inputs."""

    def test_1d_output_dtype(self):
        x = jnp.linspace(0.0, 2 * jnp.pi, 128, dtype=jnp.float64)
        dx = x[1] - x[0]
        f = jnp.sin(x)
        assert f.dtype == jnp.float64

        result = Gradient(h=[dx], acc=4)(f)
        assert result.dtype == jnp.float64

    def test_1d_accuracy(self):
        x = jnp.linspace(0.0, 2 * jnp.pi, 256, dtype=jnp.float64)
        dx = x[1] - x[0]
        f = jnp.sin(x)

        result = Gradient(h=[dx], acc=4)(f, axis=0)
        expected = jnp.cos(x)
        # float64 should give much tighter error than float32
        assert jnp.allclose(result[_interior()], expected[_interior()], atol=1e-8)

    def test_2d_output_dtype(self):
        x = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float64)
        y = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float64)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        f = X ** 2 + Y ** 2

        result = Gradient(h=[dx, dy], acc=4)(f)
        assert result.dtype == jnp.float64

    def test_2d_accuracy(self):
        x = jnp.linspace(0.0, 1.0, 100, dtype=jnp.float64)
        y = jnp.linspace(0.0, 1.0, 100, dtype=jnp.float64)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        f = jnp.sin(X) * jnp.cos(Y)

        grad = Gradient(h=[dx, dy], acc=4)(f)
        sl = _interior_2d()
        expected_x = jnp.cos(X) * jnp.cos(Y)
        expected_y = -jnp.sin(X) * jnp.sin(Y)

        assert jnp.allclose(grad[0][sl], expected_x[sl], atol=1e-7)
        assert jnp.allclose(grad[1][sl], expected_y[sl], atol=1e-7)


class TestDivergenceFloat64:
    """Divergence operator with float64 inputs."""

    def test_2d_output_dtype(self):
        x = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float64)
        y = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float64)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        # Vector field F = (X, Y)  => div(F) = 2
        F = jnp.stack([X, Y], axis=0)
        result = Divergence(h=[dx, dy], acc=4)(F)
        assert result.dtype == jnp.float64

    def test_2d_accuracy(self):
        x = jnp.linspace(0.0, 1.0, 100, dtype=jnp.float64)
        y = jnp.linspace(0.0, 1.0, 100, dtype=jnp.float64)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        # F = (sin(x)*cos(y), -cos(x)*sin(y)) => div(F) = cos(x)*cos(y) + (-cos(x)*cos(y)) = 0
        # Actually: d/dx[sin(x)cos(y)] + d/dy[-cos(x)sin(y)]
        #         = cos(x)cos(y) + (-cos(x)cos(y)) = 0
        F = jnp.stack([jnp.sin(X) * jnp.cos(Y), -jnp.cos(X) * jnp.sin(Y)], axis=0)
        result = Divergence(h=[dx, dy], acc=4)(F)
        sl = _interior_2d()
        assert jnp.allclose(result[sl], 0.0, atol=1e-8)


class TestCurlFloat64:
    """Curl operator with float64 inputs (3D only)."""

    def test_output_dtype(self):
        n = 32
        x = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
        dx = x[1] - x[0]
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")

        # F = (Y, -X, 0) => curl(F) = (0, 0, -2)
        F = jnp.stack([Y, -X, jnp.zeros_like(X)], axis=0)
        result = Curl(h=[dx, dx, dx], acc=2)(F)
        assert result.dtype == jnp.float64

    def test_accuracy(self):
        n = 32
        x = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
        dx = x[1] - x[0]
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")

        # F = (Y, -X, 0) => curl(F) = (0, 0, -2)
        F = jnp.stack([Y, -X, jnp.zeros_like(X)], axis=0)
        result = Curl(h=[dx, dx, dx], acc=2)(F)
        sl = _interior_3d()
        assert jnp.allclose(result[0][sl], 0.0, atol=1e-10)
        assert jnp.allclose(result[1][sl], 0.0, atol=1e-10)
        assert jnp.allclose(result[2][sl], -2.0, atol=1e-10)


class TestLaplacianFloat64:
    """Laplacian operator with float64 inputs."""

    def test_1d_output_dtype(self):
        x = jnp.linspace(0.0, 2 * jnp.pi, 128, dtype=jnp.float64)
        dx = x[1] - x[0]
        f = jnp.sin(x)

        result = Laplacian(h=[dx], acc=4)(f)
        assert result.dtype == jnp.float64

    def test_1d_accuracy(self):
        x = jnp.linspace(0.0, 2 * jnp.pi, 256, dtype=jnp.float64)
        dx = x[1] - x[0]
        f = jnp.sin(x)

        result = Laplacian(h=[dx], acc=4)(f)
        expected = -jnp.sin(x)  # d²/dx² sin(x) = -sin(x)
        assert jnp.allclose(result[_interior()], expected[_interior()], atol=1e-8)

    def test_2d_accuracy(self):
        x = jnp.linspace(0.0, 1.0, 80, dtype=jnp.float64)
        y = jnp.linspace(0.0, 1.0, 80, dtype=jnp.float64)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        f = X ** 3 + Y ** 3  # Laplacian = 6X + 6Y
        result = Laplacian(h=[dx, dy], acc=4)(f)
        expected = 6.0 * X + 6.0 * Y
        sl = _interior_2d()
        assert jnp.allclose(result[sl], expected[sl], atol=1e-6)


class TestJacobianFloat64:
    """Jacobian operator with float64 inputs."""

    def test_output_dtype(self):
        x = jnp.linspace(0.0, 2 * jnp.pi, 128, dtype=jnp.float64)
        dx = x[1] - x[0]
        u = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)  # (nx, 2)

        result = Jacobian(h=[dx], acc=4)(u)
        assert result.dtype == jnp.float64

    def test_accuracy(self):
        x = jnp.linspace(0.0, 2 * jnp.pi, 256, dtype=jnp.float64)
        dx = x[1] - x[0]
        u = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)

        J = Jacobian(h=[dx], acc=4)(u)  # (1, nx, 2)
        sl = _interior()
        expected = jnp.stack([jnp.cos(x), -jnp.sin(x)], axis=-1)
        assert jnp.allclose(J[0, sl, :], expected[sl, :], atol=1e-8)
