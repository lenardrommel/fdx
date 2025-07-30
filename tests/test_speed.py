import time

import pytest
from jax import numpy as jnp

from fdx import Gradient


def test_1d_grad(nx, accuracy, n_repeat=100):
    x = jnp.linspace(0, 2 * jnp.pi, nx)
    dx = x[1] - x[0]
    a_coeffs = jnp.array([1.0, 2.0, 3.0, 0.5])
    f = jnp.sum(jnp.array([jnp.sin(a * x) for a in a_coeffs]), axis=0)
    expected_grad = jnp.sum(jnp.array([a * jnp.cos(a * x) for a in a_coeffs]), axis=0)

    times = []
    errors = []
    for _ in range(n_repeat):
        start_time = time.time()
        grad = Gradient(h=[dx], acc=accuracy)
        actual_grad = grad(f, axis=0)
        times.append(time.time() - start_time)

        max_error = jnp.max(jnp.abs(actual_grad - expected_grad))
        errors.append(max_error)

    times = jnp.array(times).mean()
    errors = jnp.array(errors).mean()
    assert errors < 1e-2, (
        f"Max error {errors} exceeds tolerance for nx={nx}, accuracy={accuracy}"
    )
    print(
        f"1D Gradient: nx={nx}, accuracy={accuracy}, max error={errors}, time={times:.6f}s"
    )
    if times > 0.01:
        print("Warning: Time exceeded 0.01 seconds, consider optimizing.")
    else:
        print(f"Set new time limit for nx={nx}, accuracy={accuracy} to {times:.6f}s.")


def test_2d_grad(nx, accuracy, n_repeat=100):
    x = jnp.linspace(0, 1, 20)
    y = jnp.linspace(0, 1, 20)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    f = 2 * X**2 + Y**2  # Define a 2D scalar function: f(x,y) = 2*x^2 + y^2
    expected_grad_x = 4 * X
    expected_grad_y = 2 * Y
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    times = []
    errors = []
    for _ in range(n_repeat):
        start_time = time.time()
        grad = Gradient(h=[dx, dy], acc=accuracy)
        actual_grad = grad(f)
        times.append(time.time() - start_time)

        max_error_x = jnp.max(jnp.abs(actual_grad[0] - expected_grad_x))
        max_error_y = jnp.max(jnp.abs(actual_grad[1] - expected_grad_y))
        errors.append(max_error_x)
        errors.append(max_error_y)

    times = jnp.array(times).mean()
    errors = jnp.array(errors).mean()
    assert errors < 1e-2, (
        f"Max error {errors} exceeds tolerance for nx={nx}, accuracy={accuracy}"
    )
    print(
        f"2D Gradient: nx={nx}, accuracy={accuracy}, max error={errors}, time={times:.6f}s"
    )
    if times > 0.06:
        print("Warning: Time exceeded 0.06 seconds, consider optimizing.")
    else:
        print(f"Set new time limit for nx={nx}, accuracy={accuracy} to {times:.6f}s.")


test_2d_grad(1000, 4, n_repeat=10)
