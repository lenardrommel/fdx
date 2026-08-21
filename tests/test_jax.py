"""JAX transform compatibility tests for vector operators."""

import jax.numpy as jnp
import pytest
from jax import grad, jit, jvp, vmap

from fdx import Curl, Divergence, Gradient, Jacobian


@pytest.fixture
def operator_2d(request):
    if request.param == "gradient":
        return Gradient(h=[1.0, 1.0], acc=2)
    if request.param == "jacobian":
        return Jacobian(h=[1.0, 1.0], acc=2)
    if request.param == "divergence":
        return Divergence(h=[1.0, 1.0], acc=2)
    raise ValueError(f"Unknown operator: {request.param}")


class TestJitAndVmap:
    @pytest.mark.parametrize(
        "operator_2d,input_shape,expected_shape",
        [
            ("gradient", (10, 10), (2, 10, 10)),
            ("jacobian", (10, 10), (2, 10, 10)),
            ("divergence", (2, 10, 10), (10, 10)),
        ],
        indirect=["operator_2d"],
    )
    def test_jit_basic_shape(self, operator_2d, input_shape, expected_shape):
        field = jnp.ones(input_shape)
        out = jit(operator_2d)(field)
        assert out.shape == expected_shape

    @pytest.mark.parametrize("operator_2d", ["gradient", "jacobian"], indirect=True)
    def test_vmap_on_batched_scalar_field(self, operator_2d):
        field = jnp.ones((5, 10, 10))
        out = vmap(operator_2d)(field)
        assert out.shape == (5, 2, 10, 10)

    @pytest.mark.parametrize("operator_2d", ["divergence"], indirect=True)
    def test_vmap_on_batched_vector_field(self, operator_2d):
        field = jnp.ones((5, 2, 10, 10))
        out = vmap(operator_2d)(field)
        assert out.shape == (5, 10, 10)

    def test_vmap_jacobian_on_vector_field(self):
        operator = Jacobian(h=[1.0, 1.0], acc=2)
        field = jnp.ones((3, 10, 10, 2))
        out = vmap(operator)(field)
        assert out.shape == (3, 2, 10, 10, 2)

    def test_jit_curl_shape(self):
        operator = Curl(h=[1.0, 1.0, 1.0], acc=2)
        field = jnp.ones((3, 8, 8, 8))
        out = jit(operator)(field)
        assert out.shape == (3, 8, 8, 8)


class TestAutodiff:
    @pytest.mark.parametrize(
        "operator,field_shape,expected_shape",
        [
            (Gradient(h=[1.0, 1.0], acc=2), (10, 10), (2, 10, 10)),
            (Divergence(h=[1.0, 1.0], acc=2), (2, 10, 10), (10, 10)),
        ],
    )
    def test_jvp_linearity(self, operator, field_shape, expected_shape):
        primal = jnp.ones(field_shape)
        tangent = jnp.ones(field_shape)
        primal_out, tangent_out = jvp(operator, (primal,), (tangent,))
        assert primal_out.shape == expected_shape
        assert tangent_out.shape == expected_shape
        assert jnp.allclose(tangent_out, operator(tangent))

    @pytest.mark.parametrize(
        "operator,field",
        [
            (
                Gradient(h=[1.0, 1.0], acc=2),
                jnp.linspace(0.0, 1.0, 100, dtype=jnp.float64).reshape(10, 10),
            ),
            (Divergence(h=[1.0, 1.0], acc=2), jnp.ones((2, 10, 10), dtype=jnp.float64)),
        ],
    )
    def test_grad_and_jit_grad_agree(self, operator, field):
        def loss(u):
            out = operator(u)
            return 0.5 * jnp.sum(out**2)

        grad_eager = grad(loss)(field)
        grad_jitted = jit(grad(loss))(field)
        assert grad_eager.shape == field.shape
        assert jnp.allclose(grad_eager, grad_jitted)
