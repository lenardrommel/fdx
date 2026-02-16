"""Tests for custom_jvp on FD operators — verify JAX AD works correctly."""

import jax
import jax.numpy as jnp
import pytest

from fdx import Diff
from fdx.grids import EquidistantAxis, NonEquidistantAxis


class TestJVP:
    """Verify jax.jvp computes (D(f), D(f_dot)) for linear FD operators."""

    def test_jvp_uniform(self):
        x = jnp.linspace(0, 2 * jnp.pi, 200)
        dx = x[1] - x[0]
        d = Diff(0, EquidistantAxis(0, dx), acc=4)

        f = jnp.sin(x)
        f_dot = jnp.cos(x)

        primals, tangents = jax.jvp(d, (f,), (f_dot,))

        # primals = D(sin(x)) ≈ cos(x)
        # tangents = D(cos(x)) ≈ -sin(x)
        sl = slice(3, -3)
        assert jnp.allclose(primals[sl], jnp.cos(x)[sl], atol=1e-5)
        assert jnp.allclose(tangents[sl], -jnp.sin(x)[sl], atol=1e-5)

    def test_jvp_periodic(self):
        x = jnp.linspace(0, 2 * jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]
        d = Diff(0, EquidistantAxis(0, dx, periodic=True), acc=4)

        f = jnp.sin(x)
        f_dot = jnp.cos(x)

        primals, tangents = jax.jvp(d, (f,), (f_dot,))

        assert jnp.allclose(primals, jnp.cos(x), atol=1e-4)
        assert jnp.allclose(tangents, -jnp.sin(x), atol=1e-4)

    def test_jvp_nonuniform(self):
        x = jnp.sort(jax.random.uniform(jax.random.PRNGKey(0), (50,)) * 2 * jnp.pi)
        d = Diff(0, NonEquidistantAxis(0, x), acc=2)

        f = jnp.sin(x)
        f_dot = jnp.cos(x)

        primals, tangents = jax.jvp(d, (f,), (f_dot,))

        assert jnp.allclose(primals, jnp.cos(x), atol=0.1)
        assert jnp.allclose(tangents, -jnp.sin(x), atol=0.1)


class TestGradThroughFD:
    """Verify jax.grad works through FD operators for scalar losses."""

    def test_grad_l2_loss(self):
        """grad of ||D(f) - target||^2 w.r.t. f should be 2 * D^T * (D(f) - target)."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]
        d = Diff(0, EquidistantAxis(0, dx), acc=4)
        target = jnp.cos(x)

        def loss(f):
            return jnp.sum((d(f) - target) ** 2)

        f0 = jnp.sin(x)
        g = jax.grad(loss)(f0)

        # g should be finite and non-zero
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0)

        # Check gradient points in the right direction: loss should decrease
        lr = 1e-4
        f1 = f0 - lr * g
        assert loss(f1) < loss(f0)

    def test_grad_periodic(self):
        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        dx = x[1] - x[0]
        d = Diff(0, EquidistantAxis(0, dx, periodic=True), acc=4)

        def loss(f):
            return jnp.sum(d(f) ** 2)

        f0 = jnp.sin(x)
        g = jax.grad(loss)(f0)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0)

    def test_grad_jit_compatible(self):
        """grad + jit should work together."""
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        dx = x[1] - x[0]
        d = Diff(0, EquidistantAxis(0, dx), acc=4)

        @jax.jit
        def grad_loss(f):
            return jax.grad(lambda ff: jnp.sum(d(ff) ** 2))(f)

        f0 = jnp.sin(x)
        g = grad_loss(f0)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0)


class TestJacfwd:
    """Verify jacfwd of FD operator approximates the operator matrix."""

    def test_jacfwd_matches_matrix(self):
        n = 20
        x = jnp.linspace(0, 1, n)
        dx = x[1] - x[0]
        d = Diff(0, EquidistantAxis(0, dx), acc=2)

        J = jax.jacfwd(d)(jnp.zeros(n))
        M = d.differentiator.matrix((n,))

        assert jnp.allclose(J, M, atol=1e-10)
