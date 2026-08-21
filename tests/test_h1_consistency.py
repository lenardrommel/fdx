"""H1 inner-product consistency tests."""

import jax
import jax.numpy as jnp
import pytest

from fdx import Gradient, Jacobian


def _make_h1_with_gradient(dx, dy, acc=4):
    grad = Gradient(h=[dx, dy], acc=acc)

    def h1(u, v):
        l2 = jnp.mean(u * v)
        b, t, nx, ny, c = u.shape
        u_flat = jnp.moveaxis(u, -1, 2).reshape(b * t * c, nx, ny)
        v_flat = jnp.moveaxis(v, -1, 2).reshape(b * t * c, nx, ny)
        dux = grad(u_flat, axis=0, has_batch=True)
        duy = grad(u_flat, axis=1, has_batch=True)
        dvx = grad(v_flat, axis=0, has_batch=True)
        dvy = grad(v_flat, axis=1, has_batch=True)
        h1_semi = jnp.mean(dux * dvx) + jnp.mean(duy * dvy)
        return l2 + h1_semi

    return jax.jit(h1)


def _make_h1_with_jacobian(dx, dy, acc=4):
    jac = Jacobian(h=[dx, dy], acc=acc)

    def h1(u, v):
        l2 = jnp.mean(u * v)
        b, t, nx, ny, c = u.shape
        u_bt = u.reshape(b * t, nx, ny, c)
        v_bt = v.reshape(b * t, nx, ny, c)
        ju = jac(u_bt, has_batch=True)
        jv = jac(v_bt, has_batch=True)
        h1_semi = jnp.mean(ju[:, 0] * jv[:, 0]) + jnp.mean(ju[:, 1] * jv[:, 1])
        return l2 + h1_semi

    return jax.jit(h1)


def _make_h1_with_fft(nx, ny, dx, dy):
    kx = (2 * jnp.pi) * jnp.fft.fftfreq(nx, d=dx)
    ky = (2 * jnp.pi) * jnp.fft.rfftfreq(ny, d=dy)
    kx_b = kx.reshape(1, 1, nx, 1, 1)
    ky_b = ky.reshape(1, 1, 1, ky.shape[0], 1)

    def h1(u, v):
        l2 = jnp.mean(u * v)
        u_hat = jnp.fft.rfft2(u, axes=(2, 3))
        v_hat = jnp.fft.rfft2(v, axes=(2, 3))
        dux = jnp.fft.irfft2((1j * kx_b) * u_hat, s=(nx, ny), axes=(2, 3))
        dvx = jnp.fft.irfft2((1j * kx_b) * v_hat, s=(nx, ny), axes=(2, 3))
        duy = jnp.fft.irfft2((1j * ky_b) * u_hat, s=(nx, ny), axes=(2, 3))
        dvy = jnp.fft.irfft2((1j * ky_b) * v_hat, s=(nx, ny), axes=(2, 3))
        h1_semi = jnp.mean(dux * dvx) + jnp.mean(duy * dvy)
        return l2 + h1_semi

    return jax.jit(h1)


@pytest.fixture
def random_fields():
    shape = (8, 4, 32, 32, 5)
    dx = 1.0 / (shape[2] - 1)
    dy = 1.0 / (shape[3] - 1)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    u = jax.random.normal(k1, shape)
    v = jax.random.normal(k2, shape)
    return u, v, dx, dy


@pytest.fixture
def periodic_fields():
    b, t, nx, ny, c = 2, 2, 32, 32, 3
    x = jnp.linspace(0.0, 1.0, nx, endpoint=False)
    y = jnp.linspace(0.0, 1.0, ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    base_u = jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)
    base_v = jnp.cos(4 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)
    u = jnp.tile(base_u[None, None, :, :, None], (b, t, 1, 1, c))
    v = jnp.tile(base_v[None, None, :, :, None], (b, t, 1, 1, c))
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return u, v, dx, dy


class TestH1Consistency:
    def test_gradient_and_jacobian_implementations_agree(self, random_fields):
        u, v, dx, dy = random_fields
        h1_grad = _make_h1_with_gradient(dx, dy, acc=4)
        h1_jac = _make_h1_with_jacobian(dx, dy, acc=4)

        value_grad = h1_grad(u, v)
        value_jac = h1_jac(u, v)

        assert jnp.allclose(value_grad, value_jac, rtol=1e-4, atol=1e-4)

    def test_gradient_and_fft_implementations_are_close(self, periodic_fields):
        u, v, dx, dy = periodic_fields
        nx, ny = u.shape[2], u.shape[3]
        h1_grad = _make_h1_with_gradient(dx, dy, acc=4)
        h1_fft = _make_h1_with_fft(nx, ny, dx, dy)

        value_grad = h1_grad(u, v)
        value_fft = h1_fft(u, v)

        assert jnp.allclose(value_grad, value_fft, rtol=5e-3, atol=5e-3)
