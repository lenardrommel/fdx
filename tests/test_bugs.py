import pytest
from findiff import FinDiff
from jax import numpy as jnp

import fdx


def assert_dict_almost_equal(first, second):
    for k in set(first) & set(second):
        assert first[k] == pytest.approx(second[k], rel=1.0e-8)
    # NOTE: missing item(s) should be zero
    for k in set(first) - set(second):
        assert first[k] == pytest.approx(0, abs=1.0e-8)
    for k in set(second) - set(first):
        assert 0 == pytest.approx(second[k], abs=1.0e-8)


def test_findiff_should_raise_exception_when_applied_to_unevaluated_function():
    def f(x, y):
        return 5 * x**2 - 5 * x + 10 * y**2 - 10 * y  # pragma: no cover

    d_dx = FinDiff(1, 0.01)
    with pytest.raises(ValueError):
        d_dx(f)


def test_matrix_representation_doesnt_work_for_order_greater_2_issue_24():
    x = jnp.zeros((10))
    d3_dx3 = FinDiff((0, 1, 3))
    mat = d3_dx3.matrix(x.shape)

    assert pytest.approx(mat[0, 0]) == -2.5
    assert pytest.approx(mat[1, 1]) == -2.5
    assert pytest.approx(mat[2, 0]) == -0.5


def test_high_accuracy_results_in_type_error():
    fdx.coefficients(deriv=1, acc=16)


def test_matrix_repr_with_different_accs():
    # issue 28
    shape = (11,)
    d1 = fdx.FinDiff(0, 1, 2).matrix(shape)
    d2 = fdx.FinDiff(0, 1, 2, acc=4).matrix(shape)

    assert jnp.max(jnp.abs((d1 - d2))) > 1

    x = jnp.linspace(0, 10, 11)
    f = x**2
    df = d2.dot(f)
    assert jnp.allclose(2 * jnp.ones_like(f), df, atol=1e-3)


def test_accuracy_should_be_passed_down_to_stencil():
    shape = 11, 11
    dx = 1.0
    d1x = FinDiff(0, dx, 1, acc=4)
    stencil1 = d1x.stencil(shape)

    expected = {
        ("L", "L"): {
            (0, 0): -2.083333333333331,
            (1, 0): 3.9999999999999916,
            (2, 0): -2.999999999999989,
            (3, 0): 1.3333333333333268,
            (4, 0): -0.24999999999999858,
        },
        ("L", "C"): {
            (0, 0): -2.083333333333331,
            (1, 0): 3.9999999999999916,
            (2, 0): -2.999999999999989,
            (3, 0): 1.3333333333333268,
            (4, 0): -0.24999999999999858,
        },
        ("L", "H"): {
            (0, 0): -2.083333333333331,
            (1, 0): 3.9999999999999916,
            (2, 0): -2.999999999999989,
            (3, 0): 1.3333333333333268,
            (4, 0): -0.24999999999999858,
        },
        ("C", "L"): {
            (-2, 0): 0.08333333333333333,
            (-1, 0): -0.6666666666666666,
            (1, 0): 0.6666666666666666,
            (2, 0): -0.08333333333333333,
        },
        ("C", "C"): {
            (-2, 0): 0.08333333333333333,
            (-1, 0): -0.6666666666666666,
            (1, 0): 0.6666666666666666,
            (2, 0): -0.08333333333333333,
        },
        ("C", "H"): {
            (-2, 0): 0.08333333333333333,
            (-1, 0): -0.6666666666666666,
            (1, 0): 0.6666666666666666,
            (2, 0): -0.08333333333333333,
        },
        ("H", "L"): {
            (-4, 0): 0.24999999999999958,
            (-3, 0): -1.3333333333333313,
            (-2, 0): 2.9999999999999956,
            (-1, 0): -3.999999999999996,
            (0, 0): 2.0833333333333317,
        },
        ("H", "C"): {
            (-4, 0): 0.24999999999999958,
            (-3, 0): -1.3333333333333313,
            (-2, 0): 2.9999999999999956,
            (-1, 0): -3.999999999999996,
            (0, 0): 2.0833333333333317,
        },
        ("H", "H"): {
            (-4, 0): 0.24999999999999958,
            (-3, 0): -1.3333333333333313,
            (-2, 0): 2.9999999999999956,
            (-1, 0): -3.999999999999996,
            (0, 0): 2.0833333333333317,
        },
    }

    for char_pt in stencil1.data:
        stl = stencil1.data[char_pt]
        assert_dict_almost_equal(expected[char_pt], stl)

    d1x = FinDiff(0, dx, 1, acc=4)
    stencil1 = d1x.stencil(shape)
    for char_pt in stencil1.data:
        stl = stencil1.data[char_pt]
        assert_dict_almost_equal(expected[char_pt], stl)


def test_order_as_numpy_integer():
    order = jnp.ones(3, dtype=jnp.int32)[0]
    d_dx = FinDiff(0, 0.1, order)  # raised an AssertionError with the bug

    jnp.allclose(d_dx(jnp.linspace(0, 1, 11)), jnp.ones(11))


test_findiff_should_raise_exception_when_applied_to_unevaluated_function()
test_matrix_representation_doesnt_work_for_order_greater_2_issue_24()
test_high_accuracy_results_in_type_error()
test_matrix_repr_with_different_accs()
test_accuracy_should_be_passed_down_to_stencil()
test_order_as_numpy_integer()
