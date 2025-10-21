"""Advanced property-based tests using Hypothesis."""

import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from fdx import Curl, Diff, Divergence, FinDiff, Gradient, Laplacian
from fdx.grids import EquidistantAxis


# Custom strategies for numerical values
positive_floats = st.floats(
    min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False
)
small_positive_floats = st.floats(
    min_value=0.01, max_value=2, allow_nan=False, allow_infinity=False
)
frequencies = st.floats(
    min_value=0.5, max_value=5, allow_nan=False, allow_infinity=False
)
polynomial_powers = st.integers(min_value=1, max_value=4)
accuracy_orders = st.sampled_from([2, 4, 6])


class TestDifferentialOperatorProperties:
    """Property-based tests for differential operators."""

    @settings(deadline=None, max_examples=30)
    @given(
        coeff=small_positive_floats,
        power=polynomial_powers,
        acc=accuracy_orders,
    )
    def test_linearity_derivative(self, coeff, power, acc):
        """Property: D(cf) = c D(f) for scalar c."""
        x = jnp.linspace(0.1, 2, 50)
        dx = x[1] - x[0]

        f = x**power
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=acc)

        # Test linearity
        result1 = d_dx(coeff * f)
        result2 = coeff * d_dx(f)

        assert jnp.allclose(result1, result2, rtol=1e-4)

    @settings(deadline=None, max_examples=30)
    @given(
        a=small_positive_floats,
        b=small_positive_floats,
        power=polynomial_powers,
    )
    def test_additivity_derivative(self, a, b, power):
        """Property: D(f + g) = D(f) + D(g)."""
        x = jnp.linspace(0.1, 2, 50)
        dx = x[1] - x[0]

        f = a * x**power
        g = b * x ** (power + 1)

        d_dx = Diff(0, EquidistantAxis(0, dx), acc=6)

        result1 = d_dx(f + g)
        result2 = d_dx(f) + d_dx(g)

        assert jnp.allclose(result1, result2, rtol=1e-4)

    @settings(deadline=None, max_examples=30)
    @given(power=st.integers(min_value=2, max_value=4))
    def test_power_rule(self, power):
        """Property: d/dx(x^n) = n*x^(n-1)."""
        x = jnp.linspace(0.1, 2, 60)
        dx = x[1] - x[0]

        f = x**power
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=6)

        actual = d_dx(f)
        expected = power * x ** (power - 1)

        assert jnp.allclose(actual, expected, rtol=1e-3)

    @settings(deadline=None, max_examples=30)
    @given(freq=frequencies)
    def test_sine_derivative(self, freq):
        """Property: d/dx(sin(kx)) = k*cos(kx)."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]

        f = jnp.sin(freq * x)
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=6)

        actual = d_dx(f)
        expected = freq * jnp.cos(freq * x)

        assert jnp.allclose(actual, expected, rtol=1e-3, atol=1e-5)

    @settings(deadline=None, max_examples=30)
    @given(freq=frequencies)
    def test_cosine_derivative(self, freq):
        """Property: d/dx(cos(kx)) = -k*sin(kx)."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]

        f = jnp.cos(freq * x)
        d_dx = Diff(0, EquidistantAxis(0, dx), acc=6)

        actual = d_dx(f)
        expected = -freq * jnp.sin(freq * x)

        assert jnp.allclose(actual, expected, rtol=1e-3, atol=1e-5)


class TestGradientProperties:
    """Property-based tests for Gradient operator."""

    @settings(deadline=None, max_examples=30)
    @given(a=small_positive_floats, b=small_positive_floats)
    def test_gradient_linearity(self, a, b):
        """Property: grad(af + bg) = a*grad(f) + b*grad(g)."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = X**2
        g = Y**2

        grad = Gradient(h=[dx, dy], acc=6)

        result1 = grad(a * f + b * g)
        result2 = a * grad(f) + b * grad(g)

        assert jnp.allclose(result1, result2, rtol=1e-4)

    @settings(deadline=None, max_examples=30)
    @given(c=small_positive_floats)
    def test_gradient_constant_is_zero(self, c):
        """Property: gradient of constant is zero."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = c * jnp.ones_like(X)
        grad = Gradient(h=[dx, dy], acc=4)
        result = grad(f)

        assert jnp.allclose(result, 0, atol=1e-10)

    @settings(deadline=None, max_examples=30)
    @given(
        a=small_positive_floats,
        b=small_positive_floats,
    )
    def test_gradient_2d_quadratic(self, a, b):
        """Property: grad(ax² + by²) = [2ax, 2by]."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = a * X**2 + b * Y**2
        grad = Gradient(h=[dx, dy], acc=6)
        result = grad(f)

        expected_x = 2 * a * X
        expected_y = 2 * b * Y

        assert jnp.allclose(result[0], expected_x, rtol=1e-4, atol=1e-6)
        assert jnp.allclose(result[1], expected_y, rtol=1e-4, atol=1e-6)


class TestDivergenceProperties:
    """Property-based tests for Divergence operator."""

    @settings(deadline=None, max_examples=30)
    @given(a=small_positive_floats, b=small_positive_floats)
    def test_divergence_linear_field(self, a, b):
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
    @given(c=small_positive_floats)
    def test_divergence_linearity(self, c):
        """Property: div(cF) = c*div(F)."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        F = jnp.stack([X**2, Y**2], axis=0)
        div = Divergence(h=[dx, dy], acc=6)

        result1 = div(c * F)
        result2 = c * div(F)

        assert jnp.allclose(result1, result2, rtol=1e-4)


class TestLaplacianProperties:
    """Property-based tests for Laplacian operator."""

    @settings(deadline=None, max_examples=30)
    @given(a=small_positive_floats)
    def test_laplacian_linearity(self, a):
        """Property: Δ(af) = a*Δ(f)."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = X**2 + Y**2
        lap = Laplacian(h=[dx, dy], acc=6)

        result1 = lap(a * f)
        result2 = a * lap(f)

        assert jnp.allclose(result1, result2, rtol=1e-4)

    @settings(deadline=None, max_examples=30)
    @given(a=small_positive_floats, b=small_positive_floats)
    def test_laplacian_additivity(self, a, b):
        """Property: Δ(f + g) = Δ(f) + Δ(g)."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = a * X**2
        g = b * Y**2
        lap = Laplacian(h=[dx, dy], acc=6)

        result1 = lap(f + g)
        result2 = lap(f) + lap(g)

        assert jnp.allclose(result1, result2, rtol=1e-4)

    @settings(deadline=None, max_examples=30)
    @given(a=small_positive_floats)
    def test_laplacian_quadratic_2d(self, a):
        """Property: Δ(a(x² + y²)) = 4a."""
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


class TestCurlProperties:
    """Property-based tests for Curl operator."""

    @settings(deadline=None, max_examples=30)
    @given(a=small_positive_floats, b=small_positive_floats)
    def test_curl_gradient_field_is_zero(self, a, b):
        """Property: curl(grad(f)) = 0."""
        x = jnp.linspace(-1, 1, 20)
        y = jnp.linspace(-1, 1, 20)
        z = jnp.linspace(-1, 1, 20)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        # Gradient field of scalar function
        F = jnp.stack([a * X, b * Y, a * Z], axis=0)

        curl = Curl(h=[dx, dy, dz], acc=4)
        result = curl(F)

        # Curl of gradient should be zero
        assert jnp.allclose(result, 0, atol=1e-6)

    @settings(deadline=None, max_examples=30)
    @given(c=small_positive_floats)
    def test_curl_linearity(self, c):
        """Property: curl(cF) = c*curl(F)."""
        x = jnp.linspace(-1, 1, 20)
        y = jnp.linspace(-1, 1, 20)
        z = jnp.linspace(-1, 1, 20)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        F = jnp.stack([-Y, X, jnp.zeros_like(Z)], axis=0)
        curl = Curl(h=[dx, dy, dz], acc=4)

        result1 = curl(c * F)
        result2 = c * curl(F)

        assert jnp.allclose(result1, result2, rtol=1e-4)


class TestFinDiffProperties:
    """Property-based tests for FinDiff compatibility layer."""

    @settings(deadline=None, max_examples=30)
    @given(
        a=small_positive_floats,
        power=polynomial_powers,
    )
    def test_findiff_polynomial(self, a, power):
        """Property: d/dx(ax^n) = anx^(n-1)."""
        x = jnp.linspace(0.1, 2, 50)
        dx = x[1] - x[0]

        f = a * x**power
        d_dx = FinDiff(0, dx, acc=6)

        actual = d_dx(f)
        expected = a * power * x ** (power - 1)

        assert jnp.allclose(actual, expected, rtol=1e-3)

    @settings(deadline=None, max_examples=30)
    @given(freq=frequencies)
    def test_findiff_sine(self, freq):
        """Property: d/dx(sin(kx)) = k*cos(kx)."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]

        f = jnp.sin(freq * x)
        d_dx = FinDiff(0, dx, acc=6)

        actual = d_dx(f)
        expected = freq * jnp.cos(freq * x)

        assert jnp.allclose(actual, expected, rtol=1e-3, atol=1e-5)


class TestVectorCalculusIdentities:
    """Tests for vector calculus identities."""

    def test_divergence_of_curl_is_zero(self):
        """Identity: div(curl(F)) = 0."""
        x = jnp.linspace(-1, 1, 25)
        y = jnp.linspace(-1, 1, 25)
        z = jnp.linspace(-1, 1, 25)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        # Some vector field
        F = jnp.stack([Y * Z, X * Z, X * Y], axis=0)

        curl = Curl(h=[dx, dy, dz], acc=4)
        div = Divergence(h=[dx, dy, dz], acc=4)

        curl_F = curl(F)
        result = div(curl_F)

        # div(curl(F)) should be zero
        assert jnp.allclose(result, 0, atol=1e-5)

    @given(a=small_positive_floats, b=small_positive_floats, c=small_positive_floats)
    def test_laplacian_divergence_gradient(self, a, b, c):
        """Identity: Δf = div(grad(f))."""
        x = jnp.linspace(0, 1, 30)
        y = jnp.linspace(0, 1, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        f = a * X**2 + b * Y**2 + c

        lap = Laplacian(h=[dx, dy], acc=6)
        grad = Gradient(h=[dx, dy], acc=6)
        div = Divergence(h=[dx, dy], acc=6)

        result1 = lap(f)
        result2 = div(grad(f))

        assert jnp.allclose(result1, result2, rtol=1e-3, atol=1e-6)
