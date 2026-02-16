"""JIT smoke tests — verify operators work inside jax.jit."""

import jax
import jax.numpy as jnp
import pytest

from fdx import Diff
from fdx.grids import EquidistantAxis, NonEquidistantAxis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uniform_diff(n=100, acc=4):
    """Return (Diff, x, dx) on a uniform grid."""
    x = jnp.linspace(0, 2 * jnp.pi, n)
    dx = x[1] - x[0]
    d = Diff(0, EquidistantAxis(0, dx), acc=acc)
    return d, x, dx


def _make_nonuniform_diff(n=50, acc=2):
    """Return (Diff, x) on a non-uniform grid."""
    x = jnp.sort(jax.random.uniform(jax.random.PRNGKey(42), (n,)) * 2 * jnp.pi)
    d = Diff(0, NonEquidistantAxis(0, x), acc=acc)
    return d, x


def _make_periodic_diff(n=100, acc=4):
    """Return (Diff, x) on a periodic uniform grid."""
    x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    dx = x[1] - x[0]
    d = Diff(0, EquidistantAxis(0, dx, periodic=True), acc=acc)
    return d, x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJitUniform:
    """JIT with uniform grid Diff."""

    def test_jit_uniform_first_derivative(self):
        d, x, dx = _make_uniform_diff()
        f = jnp.sin(x)

        @jax.jit
        def apply(op, y):
            return op(y)

        result = apply(d, f)
        expected = jnp.cos(x)
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_jit_uniform_second_derivative(self):
        x = jnp.linspace(0, 2 * jnp.pi, 200)
        dx = x[1] - x[0]
        d = Diff(0, EquidistantAxis(0, dx), acc=4)
        d2 = d ** 2
        f = jnp.sin(x)

        @jax.jit
        def apply(op, y):
            return op(y)

        result = apply(d2, f)
        expected = -jnp.sin(x)
        assert jnp.allclose(result, expected, atol=1e-3)


class TestJitNonUniform:
    """JIT with non-uniform grid Diff."""

    def test_jit_nonuniform_first_derivative(self):
        d, x = _make_nonuniform_diff()
        f = jnp.sin(x)

        @jax.jit
        def apply(op, y):
            return op(y)

        result = apply(d, f)
        expected = jnp.cos(x)
        assert jnp.allclose(result, expected, atol=0.1)


class TestJitPeriodic:
    """JIT with periodic uniform grid Diff."""

    def test_jit_periodic_first_derivative(self):
        d, x = _make_periodic_diff()
        f = jnp.sin(x)

        @jax.jit
        def apply(op, y):
            return op(y)

        result = apply(d, f)
        expected = jnp.cos(x)
        assert jnp.allclose(result, expected, atol=1e-4)


class TestJitComposition:
    """JIT with composed operators."""

    def test_jit_multiply_composition(self):
        d, x, dx = _make_uniform_diff()
        d_composed = d * d  # second derivative via composition
        f = jnp.sin(x)

        @jax.jit
        def apply(op, y):
            return op(y)

        result = apply(d_composed, f)
        expected = -jnp.sin(x)
        assert jnp.allclose(result, expected, atol=1e-3)


class TestPytreeRoundtrip:
    """Verify pytree flatten/unflatten preserves operator semantics."""

    def test_uniform_roundtrip(self):
        d, x, dx = _make_uniform_diff()
        f = jnp.sin(x)

        leaves, treedef = jax.tree_util.tree_flatten(d)
        d2 = jax.tree_util.tree_unflatten(treedef, leaves)

        result_orig = d(f)
        result_rt = d2(f)
        assert jnp.allclose(result_orig, result_rt)

    def test_nonuniform_roundtrip(self):
        d, x = _make_nonuniform_diff()
        f = jnp.sin(x)

        leaves, treedef = jax.tree_util.tree_flatten(d)
        d2 = jax.tree_util.tree_unflatten(treedef, leaves)

        result_orig = d(f)
        result_rt = d2(f)
        assert jnp.allclose(result_orig, result_rt)

    def test_periodic_roundtrip(self):
        d, x = _make_periodic_diff()
        f = jnp.sin(x)

        leaves, treedef = jax.tree_util.tree_flatten(d)
        d2 = jax.tree_util.tree_unflatten(treedef, leaves)

        result_orig = d(f)
        result_rt = d2(f)
        assert jnp.allclose(result_orig, result_rt)
