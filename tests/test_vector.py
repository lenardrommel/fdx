# test_vector.py

import pytest
from jax import numpy as jnp

from fdx import Gradient, Jacobian


def _interior_1d(width: int = 2) -> slice:
    return slice(width, -width)


def _interior_2d(width: int = 2) -> tuple[slice, slice]:
    return (slice(width, -width), slice(width, -width))


def _fd_atol(dx: float, acc: int, C: float = 10.0) -> float:
    """
    Calculates absolute tolerance based on grid spacing and accuracy order.
    atol ~ O(dx^p) where p is the accuracy order.
    """
    p = 2 if acc <= 2 else acc
    return C * float(dx) ** p


def _relative_l2_error(actual: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-12) -> float:
    num = jnp.linalg.norm(actual - target)
    den = jnp.linalg.norm(target)
    return float(num / (den + eps))


def test_gradient_1d_sine_axis_derivative():
    x = jnp.linspace(0.0, 2.0 * jnp.pi, 256)
    dx = x[1] - x[0]
    acc = 4
    f = jnp.sin(x)
    target = jnp.cos(x)

    # Use Gradient with specific axis
    grad_op = Gradient(h=[dx], acc=acc)
    # axis=0 is the only axis here
    df = grad_op(f, axis=0)

    sl = _interior_1d(width=2)
    # df should be shape (nx,)
    rel_err = _relative_l2_error(df[sl], target[sl])

    atol = _fd_atol(dx, acc, C=5.0)
    # Using L2 relative error check
    assert rel_err < atol  # Strictly speaking relative error isn't atol, but atol
    # scales with dx^p which is what we want for truncation error relative to order 1 signal


def test_gradient_2d_polynomial_full_gradient():
    nx, ny = 120, 96
    x = jnp.linspace(0.0, 2.0, nx)
    y = jnp.linspace(-1.0, 1.0, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    acc = 4

    f = X**3 + 2.0 * X * Y + Y**2
    grad_f = Gradient(h=[dx, dy], acc=acc)(f, axis=None)
    # grad_f shape: (2, nx, ny)

    target_x = 3.0 * X**2 + 2.0 * Y
    target_y = 2.0 * X + 2.0 * Y
    sl = _interior_2d(width=2)

    atol = max(_fd_atol(dx, acc), _fd_atol(dy, acc))
    rtol = 1e-3

    # Check x-component
    assert jnp.allclose(grad_f[0][sl], target_x[sl], rtol=rtol, atol=atol)
    # Check y-component
    assert jnp.allclose(grad_f[1][sl], target_y[sl], rtol=rtol, atol=atol)


@pytest.mark.parametrize("axis", [0, 1])
def test_gradient_axis_uses_correct_spacing_regression(axis):
    # Distinct spacings to catch axis swaps
    x = jnp.linspace(0.0, 1.0, 101)  # dx ~ 0.01
    y = jnp.linspace(0.0, 2.0, 51)   # dy ~ 0.04
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = jnp.meshgrid(x, y, indexing="ij")
    # Function depends only on one axis
    if axis == 0:
        f = jnp.sin(X)
        target = jnp.cos(X)
    else:
        f = jnp.sin(Y)
        target = jnp.cos(Y)

    # Gradient along specific axis
    df = Gradient(h=[dx, dy], acc=4)(f, axis=axis)

    sl = _interior_2d(width=2)
    relevant_dx = dx if axis == 0 else dy
    atol = _fd_atol(relevant_dx, 4, C=10.0)
    rtol = 1e-3

    assert jnp.allclose(df[sl], target[sl], rtol=rtol, atol=atol)


def test_gradient_has_batch_full():
    x = jnp.linspace(0.0, 1.0, 100)
    dx = x[1] - x[0]
    B = 7
    # Batch of scalar fields: (B, nx)
    f = jnp.stack([jnp.sin((k + 1) * x) for k in range(B)], axis=0)

    # axis=None with batch -> returns (B, ndims, nx)
    grad_op = Gradient(h=[dx], acc=4)
    g = grad_op(f, axis=None, has_batch=True)

    assert g.shape == (B, 1, x.size)

    target = jnp.stack([(k + 1) * jnp.cos((k + 1) * x) for k in range(B)], axis=0)
    sl = _interior_1d(width=2)
    atol = _fd_atol(dx, 4, C=10.0)
    rtol = 1e-3

    # Check match for component 0 (the only spatial dimension)
    assert jnp.allclose(g[:, 0, sl], target[:, sl], rtol=rtol, atol=atol)


def test_jacobian_1d_time_channels():
    nx, nt, nc = 200, 4, 3
    x = jnp.linspace(0.0, 2.0 * jnp.pi, nx)
    dx = x[1] - x[0]
    a = 3.0

    t = jnp.arange(nt, dtype=x.dtype) + 1.0
    c = jnp.arange(nc, dtype=x.dtype) + 2.0

    # u(x, t, c) = sin(ax) * t * c
    # shape (nx, nt, nc)
    u = jnp.sin(a * x)[:, None, None] * t[None, :, None] * c[None, None, :]

    # Jacobian no batch -> (ndims, nx, nt, nc)
    J = Jacobian(h=[dx], acc=4)(u)
    assert J.shape == (1, nx, nt, nc)

    target = (a * jnp.cos(a * x))[:, None, None] * t[None, :, None] * c[None, None, :]
    sl = _interior_1d(width=2)
    atol = _fd_atol(dx, 4, C=10.0)
    rtol = 1e-3

    assert jnp.allclose(J[0, sl, :, :], target[sl, :, :], rtol=rtol, atol=atol)


def test_jacobian_1d_with_batch():
    B, nx, nt, nc = 5, 200, 4, 2
    x = jnp.linspace(0.0, 2.0 * jnp.pi, nx)
    dx = x[1] - x[0]
    a = 2.0

    t = jnp.arange(nt, dtype=x.dtype) + 1.0
    c = jnp.arange(nc, dtype=x.dtype) + 1.0

    base = jnp.sin(a * x)[:, None, None] * t[None, :, None] * c[None, None, :]
    # u shape: (B, nx, nt, nc)
    u = jnp.stack([(k + 1) * base for k in range(B)], axis=0)

    # Jacobian with batch -> (B, ndims, nx, nt, nc)
    J = Jacobian(h=[dx], acc=4)(u, has_batch=True)
    assert J.shape == (B, 1, nx, nt, nc)

    target_base = (a * jnp.cos(a * x))[:, None, None] * t[None, :, None] * c[None, None, :]
    target = jnp.stack([(k + 1) * target_base for k in range(B)], axis=0)

    sl = _interior_1d(width=2)
    atol = _fd_atol(dx, 4, C=10.0)
    rtol = 1e-3

    assert jnp.allclose(J[:, 0, sl, :, :], target[:, sl, :, :], rtol=rtol, atol=atol)


def test_jacobian_2d_vector_field_shapes_and_values():
    nx, ny = 64, 48
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # two "components" (like channels): u0 = sin(2x), u1 = cos(3y)
    # Using small coefficients to keep derivatives clean
    u = jnp.stack([jnp.sin(2*X), jnp.cos(3*Y)], axis=-1)  # (nx, ny, 2)

    J = Jacobian(h=[dx, dy], acc=4)(u)  # (2, nx, ny, 2)
    assert J.shape == (2, nx, ny, 2)

    # ∂/∂x of [sin(2x), cos(3y)] = [2cos(2x), 0]
    target_dx = jnp.stack([2*jnp.cos(2*X), jnp.zeros_like(X)], axis=-1)
    # ∂/∂y of [sin(2x), cos(3y)] = [0, -3sin(3y)]
    target_dy = jnp.stack([jnp.zeros_like(Y), -3*jnp.sin(3*Y)], axis=-1)

    sl = _interior_2d(2)
    atol = max(_fd_atol(dx, 4), _fd_atol(dy, 4))

    # Check J[0] which is d/dx
    assert jnp.allclose(J[0][sl], target_dx[sl], rtol=1e-3, atol=atol)
    # Check J[1] which is d/dy
    assert jnp.allclose(J[1][sl], target_dy[sl], rtol=1e-3, atol=atol)


def test_jacobian_matches_gradient_for_scalar_field():
    x = jnp.linspace(0.0, 1.0, 200)
    dx = x[1] - x[0]
    f = jnp.exp(2.0 * x)

    G = Gradient(h=[dx], acc=4)(f)
    J = Jacobian(h=[dx], acc=4)(f)  # should be (1, nx)

    assert J.shape == (1, len(x))

    sl = _interior_1d(2)
    _fd_atol(dx, 4)
    # They should be exactly identical if implementation reuses components,
    # but allowing small tolerance just in case of minor ops differences
    assert jnp.allclose(J[0, sl], G[0, sl], rtol=1e-10, atol=1e-10)


def test_vector_derivatives_numerical_stability_finite_outputs():
    # Stress test with high frequency relative to grid
    x = jnp.linspace(0.0, 2.0 * jnp.pi, 512)
    dx = x[1] - x[0]
    amp = 1.0e6
    freq = 80.0 # Higher frequency to stress cancellation

    f = amp * jnp.sin(freq * x)

    grad = Gradient(h=[dx], acc=4)
    df = grad(f, axis=0)
    target_df = amp * freq * jnp.cos(freq * x)

    sl = _interior_1d(width=2)
    rel_err_grad = _relative_l2_error(df[sl], target_df[sl])

    assert bool(jnp.all(jnp.isfinite(df)))
    # With high frequency, error might be larger, so allow more slack
    # But mainly we want to ensure it doesn't blow up
    atol = _fd_atol(dx, 4, C=100.0)
    assert rel_err_grad < atol or rel_err_grad < 0.1 # Fallback for high freq

    # Check Jacobian stability
    u = jnp.stack([f, 0.5 * f], axis=-1)  # (nx, 2)
    J = Jacobian(h=[dx], acc=4)(u)        # (1, nx, 2)

    target_J = jnp.stack([target_df, 0.5 * target_df], axis=-1)

    rel_err_jac = _relative_l2_error(J[0, sl, :], target_J[sl, :])
    assert bool(jnp.all(jnp.isfinite(J)))
    assert rel_err_jac < atol or rel_err_jac < 0.1
