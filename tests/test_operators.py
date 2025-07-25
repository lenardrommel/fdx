from jax import numpy as jnp

from fdx import Diff

# def test_field_operator():
#     f = jnp.array([1.0, 2.0, 3.0])
#     S = FieldOperator(2, shape=f.shape)
#     jnp.allclose((S + S)(f), 4 * f)


def test_simple_derivatives():
    x = jnp.linspace(0, 1, 100)
    dx = x[1] - x[0]
    f = x**3

    d_dx = Diff(0, dx)

    actual = d_dx(f)

    assert jnp.allclose(actual, 3 * x**2, atol=1e-3)


# test_field_operator()
test_simple_derivatives()
