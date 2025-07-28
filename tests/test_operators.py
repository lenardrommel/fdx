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

from fdx import FinDiff

x = jnp.linspace(0, 10, 100)
dx = x[1] - x[0]
f = jnp.sin(x)
g = jnp.cos(x)

d2_dx2 = FinDiff(0, dx, 2, acc=10)
result = d2_dx2(f)
x, y, z = [jnp.linspace(0, 10, 100)] * 3
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
f = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)

d_dx = FinDiff(0, dx)
d_dz = FinDiff(2, dz)

d3_dx2dy = FinDiff((0, dx, 2), (1, dy))
result = d3_dx2dy(f)
