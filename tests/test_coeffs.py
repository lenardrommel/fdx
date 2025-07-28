from typing import Any, Dict

import jax
import jax.numpy as jnp
import pytest

# Original findiff imports for comparison
from findiff import coefficients as coefficients_np
from findiff.coefs import calc_coefs as calc_coefs_np
from findiff.coefs import coefficients_non_uni as coefficients_non_uni_np
from findiff.coefs import compute_inverse_Vandermonde as compute_inverse_Vandermonde_np
from jax import Array

# Assuming these are your JAX implementations
from fdx.coefs import (
    coefficients,
    coefficients_non_uni,
    compute_coeffs,
    compute_inverse_Vandermonde,
)


def test_compute_inverse_Vandermonde():
    """Test JAX implementation against original findiff"""
    c = compute_inverse_Vandermonde(column=2, offsets=[-1, 0, 1])
    c_np = compute_inverse_Vandermonde_np(column=2, offsets=[-1, 0, 1], symbolic=False)
    assert jnp.allclose(c, c_np)


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_order2_acc2(analytic_inv):
    """Test 2nd order derivative with 2nd order accuracy"""
    c = coefficients(deriv=2, acc=2, analytic_inv=analytic_inv)
    c_np = coefficients_np(deriv=2, acc=2, analytic_inv=analytic_inv)

    # Test center coefficients
    coefs = c["center"]["coefficients"]
    coefs_np = c_np["center"]["coefficients"]
    assert jnp.allclose(coefs, jnp.array([1.0, -2.0, 1.0]))
    assert jnp.allclose(coefs, coefs_np)

    offs = c["center"]["offsets"]
    offs_np = c_np["center"]["offsets"]
    assert jnp.allclose(offs, jnp.array([-1, 0, 1]))
    assert jnp.allclose(offs, offs_np)

    # Test forward coefficients
    coefs = c["forward"]["coefficients"]
    coefs_np = c_np["forward"]["coefficients"]
    assert jnp.allclose(coefs, jnp.array([2, -5, 4, -1]))
    assert jnp.allclose(coefs, coefs_np)

    offs = c["forward"]["offsets"]
    offs_np = c_np["forward"]["offsets"]
    assert jnp.allclose(offs, jnp.array([0, 1, 2, 3]))
    assert jnp.allclose(offs, offs_np)

    # Test backward coefficients
    coefs = c["backward"]["coefficients"]
    coefs_np = c_np["backward"]["coefficients"]
    expected_backward = jnp.array([2, -5, 4, -1])[::-1]
    assert jnp.allclose(coefs, expected_backward)
    assert jnp.allclose(coefs, coefs_np)

    offs = c["backward"]["offsets"]
    offs_np = c_np["backward"]["offsets"]
    assert jnp.allclose(offs, jnp.array([-3, -2, -1, 0]))
    assert jnp.allclose(offs, offs_np)


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_order1_acc2(analytic_inv):
    """Test 1st order derivative with 2nd order accuracy"""
    c = coefficients(deriv=1, acc=2, analytic_inv=analytic_inv)
    c_np = coefficients_np(deriv=1, acc=2, analytic_inv=analytic_inv)

    # Test center coefficients
    coefs = c["center"]["coefficients"]
    coefs_np = c_np["center"]["coefficients"]
    assert jnp.allclose(coefs, jnp.array([-0.5, 0, 0.5]))
    assert jnp.allclose(coefs, coefs_np)

    offs = c["center"]["offsets"]
    offs_np = c_np["center"]["offsets"]
    assert jnp.allclose(offs, jnp.array([-1, 0, 1]))
    assert jnp.allclose(offs, offs_np)

    # Test forward coefficients
    coefs = c["forward"]["coefficients"]
    coefs_np = c_np["forward"]["coefficients"]
    assert jnp.allclose(coefs, jnp.array([-1.5, 2, -0.5]))
    assert jnp.allclose(coefs, coefs_np)

    offs = c["forward"]["offsets"]
    offs_np = c_np["forward"]["offsets"]
    assert jnp.allclose(offs, jnp.array([0, 1, 2]))
    assert jnp.allclose(offs, offs_np)

    # Test backward coefficients
    coefs = c["backward"]["coefficients"]
    coefs_np = c_np["backward"]["coefficients"]
    expected_backward = -jnp.array([-1.5, 2, -0.5])[::-1]
    assert jnp.allclose(coefs, expected_backward)
    assert jnp.allclose(coefs, coefs_np)

    offs = c["backward"]["offsets"]
    offs_np = c_np["backward"]["offsets"]
    assert jnp.allclose(offs, jnp.array([-2, -1, 0]))
    assert jnp.allclose(offs, offs_np)


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_order1_acc4(analytic_inv):
    """Test 1st order derivative with 4th order accuracy"""
    c = coefficients(deriv=1, acc=4, analytic_inv=analytic_inv)
    c_np = coefficients_np(deriv=1, acc=4, analytic_inv=analytic_inv)

    # Test center coefficients
    coefs = c["center"]["coefficients"]
    coefs_np = c_np["center"]["coefficients"]
    expected_center = jnp.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
    assert jnp.allclose(coefs, expected_center, atol=1e-4), (
        f"Expected {expected_center}, got {coefs}"
    )
    assert jnp.allclose(coefs, coefs_np, atol=1e-4), f"Expected {coefs_np}, got {coefs}"

    offs = c["center"]["offsets"]
    offs_np = c_np["center"]["offsets"]
    assert jnp.allclose(offs, jnp.array([-2, -1, 0, 1, 2]), atol=1e-4)
    assert jnp.allclose(offs, offs_np, atol=1e-4)

    # Test forward coefficients
    coefs = c["forward"]["coefficients"]
    coefs_np = c_np["forward"]["coefficients"]
    expected_forward = jnp.array([-25 / 12, 4, -3, 4 / 3, -1 / 4])
    assert jnp.allclose(coefs, expected_forward, atol=1e-4)
    assert jnp.allclose(coefs, coefs_np, atol=1e-4)

    offs = c["forward"]["offsets"]
    offs_np = c_np["forward"]["offsets"]
    assert jnp.allclose(offs, jnp.array([0, 1, 2, 3, 4]), atol=1e-4)
    assert jnp.allclose(offs, offs_np, atol=1e-4)

    # Test backward coefficients
    coefs = c["backward"]["coefficients"]
    coefs_np = c_np["backward"]["coefficients"]
    expected_backward = -jnp.array([-25 / 12, 4, -3, 4 / 3, -1 / 4])[::-1]
    assert jnp.allclose(coefs, expected_backward, atol=1e-4)
    assert jnp.allclose(coefs, coefs_np, atol=1e-4)

    offs = c["backward"]["offsets"]
    offs_np = c_np["backward"]["offsets"]
    expected_offs_backward = -jnp.array([0, 1, 2, 3, 4])[::-1]
    assert jnp.allclose(offs, expected_offs_backward, atol=1e-4)
    assert jnp.allclose(offs, offs_np, atol=1e-4)


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_order2_acc4(analytic_inv):
    """Test 2nd order derivative with 4th order accuracy"""
    c = coefficients(deriv=2, acc=4, analytic_inv=analytic_inv)
    c_np = coefficients_np(deriv=2, acc=4, analytic_inv=analytic_inv)

    # Test center coefficients
    coefs = c["center"]["coefficients"]
    coefs_np = c_np["center"]["coefficients"]
    expected_center = jnp.array([-1 / 12, 4 / 3, -2.5, 4 / 3, -1 / 12])
    assert jnp.allclose(coefs, expected_center, atol=1e-4)
    assert jnp.allclose(coefs, coefs_np, atol=1e-4)

    offs = c["center"]["offsets"]
    offs_np = c_np["center"]["offsets"]
    assert jnp.allclose(offs, jnp.array([-2, -1, 0, 1, 2]), atol=1e-4)
    assert jnp.allclose(offs, offs_np, atol=1e-4)

    # Test forward coefficients
    coefs = c["forward"]["coefficients"]
    coefs_np = c_np["forward"]["coefficients"]
    expected_forward = jnp.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6])
    assert jnp.allclose(coefs, expected_forward, atol=1e-4)
    assert jnp.allclose(coefs, coefs_np, atol=1e-4)

    offs = c["forward"]["offsets"]
    offs_np = c_np["forward"]["offsets"]
    assert jnp.allclose(offs, jnp.array([0, 1, 2, 3, 4, 5]), atol=1e-4)
    assert jnp.allclose(offs, offs_np, atol=1e-4)

    # Test backward coefficients
    coefs = c["backward"]["coefficients"]
    coefs_np = c_np["backward"]["coefficients"]
    expected_backward = jnp.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6])[
        ::-1
    ]
    assert jnp.allclose(coefs, expected_backward, atol=1e-4)
    assert jnp.allclose(coefs, coefs_np, atol=1e-4)

    offs = c["backward"]["offsets"]
    offs_np = c_np["backward"]["offsets"]
    expected_offs_backward = -jnp.array([0, 1, 2, 3, 4, 5])[::-1]
    assert jnp.allclose(offs, expected_offs_backward, atol=1e-4)
    assert jnp.allclose(offs, offs_np, atol=1e-4)


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_accuracy_central_deriv2_acc2(analytic_inv):
    """Test accuracy calculation for central difference"""
    coefs = compute_coeffs(2, [-1, 0, 1], analytic_inv=analytic_inv)
    coefs_np = calc_coefs_np(2, [-1, 0, 1], analytic_inv=analytic_inv)

    assert coefs["accuracy"] == 2
    assert coefs["accuracy"] == coefs_np["accuracy"]


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_accuracy_central_deriv1_acc2(analytic_inv):
    """Test accuracy calculation for 1st derivative central difference"""
    coefs = compute_coeffs(1, [-1, 0, 1], analytic_inv=analytic_inv)
    coefs_np = calc_coefs_np(1, [-1, 0, 1], analytic_inv=analytic_inv)

    assert coefs["accuracy"] == 2
    assert coefs["accuracy"] == coefs_np["accuracy"]


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_accuracy_left1_right0_deriv1_acc1(analytic_inv):
    """Test accuracy calculation for forward difference"""
    coefs = compute_coeffs(1, [-1, 0], analytic_inv=analytic_inv)
    coefs_np = calc_coefs_np(1, [-1, 0], analytic_inv=analytic_inv)

    assert coefs["accuracy"] == 1
    assert coefs["accuracy"] == coefs_np["accuracy"]


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_accuracy_left0_right3_deriv1_acc3(analytic_inv):
    """Test accuracy calculation for specific offset pattern"""
    coefs = compute_coeffs(2, [0, 1, 2, 3], analytic_inv=analytic_inv)
    coefs_np = calc_coefs_np(2, [0, 1, 2, 3], analytic_inv=analytic_inv)

    assert coefs["accuracy"] == 2
    assert coefs["accuracy"] == coefs_np["accuracy"]


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_accuracy_from_offsets_symbolic1(analytic_inv):
    """Test accuracy calculation with symbolic mode"""
    # Note: JAX doesn't have symbolic mode, so we test without it
    coefs = compute_coeffs(2, [0, 1, 2, 3], analytic_inv=analytic_inv)
    coefs_np = calc_coefs_np(2, [0, 1, 2, 3], symbolic=True, analytic_inv=analytic_inv)

    assert coefs["accuracy"] == 2
    assert coefs["accuracy"] == coefs_np["accuracy"]


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_accuracy_from_offsets_symbolic2(analytic_inv):
    """Test accuracy calculation with symmetric offsets"""
    coefs = compute_coeffs(2, [-4, -2, 0, 2, 4], analytic_inv=analytic_inv)
    coefs_np = calc_coefs_np(
        2, [-4, -2, 0, 2, 4], symbolic=True, analytic_inv=analytic_inv
    )

    assert coefs["accuracy"] == 4
    assert coefs["accuracy"] == coefs_np["accuracy"]


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_coefs_from_offsets(analytic_inv):
    """Test coefficient calculation from custom offsets"""
    coefs = compute_coeffs(1, [-2, 0, 1], analytic_inv=analytic_inv)
    coefs_np = calc_coefs_np(1, [-2, 0, 1], analytic_inv=analytic_inv)

    expected = jnp.array([-1.0 / 6, -0.5, 2.0 / 3])
    assert jnp.allclose(coefs["coefficients"], expected)
    assert jnp.allclose(coefs["coefficients"], coefs_np["coefficients"])


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_coefs_from_offsets_no_central_point(analytic_inv):
    """Test coefficient calculation without central point"""
    coefs = compute_coeffs(1, [-2, 1, 2], analytic_inv=analytic_inv)
    coefs_np = calc_coefs_np(1, [-2, 1, 2], analytic_inv=analytic_inv)

    expected = jnp.array([-1.0 / 4, 0, 1.0 / 4])
    assert jnp.allclose(coefs["coefficients"], expected)
    assert jnp.allclose(coefs["coefficients"], coefs_np["coefficients"])


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_coefs_from_offsets_not_enough_points(analytic_inv):
    """Test error handling for insufficient points"""
    with pytest.raises(ValueError):
        coefficients(2, offsets=[-2, 2], analytic_inv=analytic_inv)


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_calc_coefs_symbolic(analytic_inv):
    """Test symbolic coefficient calculation"""
    # JAX version without symbolic mode
    coefs = compute_coeffs(1, [-2, 0, 1], analytic_inv=analytic_inv)
    # Compare with numpy version that has symbolic capability
    coefs_np = calc_coefs_np(1, [-2, 0, 1], symbolic=False, analytic_inv=analytic_inv)

    expected = jnp.array([-1 / 6, -1 / 2, 2 / 3])
    assert jnp.allclose(coefs["coefficients"], expected)
    assert jnp.allclose(coefs["coefficients"], coefs_np["coefficients"])


@pytest.mark.parametrize("analytic_inv", [True, False])
def test_non_uniform(analytic_inv):
    """Test non-uniform grid coefficients"""
    x = jnp.linspace(0, 10, 100)
    x_np = jnp.array(x)  # Convert to numpy for original function
    dx = x[1] - x[0]

    # Uniform coefficients (JAX)
    c_uni = coefficients(deriv=2, acc=2, analytic_inv=analytic_inv)
    coefs_uni = c_uni["center"]["coefficients"] / dx**2

    # Non-uniform coefficients (JAX)
    c_non_uni = coefficients_non_uni(deriv=2, acc=2, coords=x, idx=5)
    coefs_non_uni = c_non_uni["coefficients"]

    # Compare with original implementation
    c_non_uni_np = coefficients_non_uni_np(deriv=2, acc=2, coords=x_np, idx=5)
    coefs_non_uni_np = c_non_uni_np["coefficients"]

    assert jnp.allclose(coefs_non_uni, coefs_uni)
    assert jnp.allclose(coefs_non_uni, coefs_non_uni_np)


def test_invalid_acc_raises_exception():
    """Test error handling for invalid accuracy values"""
    with pytest.raises(ValueError):
        coefficients(deriv=1, acc=3)
    with pytest.raises(ValueError):
        coefficients(deriv=1, acc=0)
    with pytest.raises(ValueError):
        coefficients_non_uni(1, 3, None, None)
    with pytest.raises(ValueError):
        coefficients_non_uni(1, 0, None, None)


def test_invalid_deriv_raises_exception():
    """Test error handling for invalid derivative orders"""
    with pytest.raises(ValueError):
        coefficients(-1, 2)
    with pytest.raises(ValueError):
        coefficients_non_uni(-1, 2, None, None)


# Convenience function to run all tests
def run_all_tests():
    """Run all tests with both analytic_inv options"""
    print("Running JAX finite difference coefficient tests...")

    test_compute_inverse_Vandermonde()
    print("✓ Vandermonde inverse test passed")

    for analytic_inv in [False, True]:
        test_order2_acc2(analytic_inv)
        test_order1_acc2(analytic_inv)
        test_order1_acc4(analytic_inv)
        test_order2_acc4(analytic_inv)
        test_calc_accuracy_central_deriv2_acc2(analytic_inv)
        test_calc_accuracy_central_deriv1_acc2(analytic_inv)
        test_calc_accuracy_left1_right0_deriv1_acc1(analytic_inv)
        test_calc_accuracy_left0_right3_deriv1_acc3(analytic_inv)
        test_calc_accuracy_from_offsets_symbolic1(analytic_inv)
        test_calc_accuracy_from_offsets_symbolic2(analytic_inv)
        test_calc_coefs_from_offsets(analytic_inv)
        test_calc_coefs_from_offsets_no_central_point(analytic_inv)
        test_calc_coefs_symbolic(analytic_inv)
        test_non_uniform(analytic_inv)

        print(f"✓ All parameterized tests passed for analytic_inv={analytic_inv}")

    test_calc_coefs_from_offsets_not_enough_points(False)
    test_invalid_acc_raises_exception()
    test_invalid_deriv_raises_exception()
    print("✓ Error handling tests passed")

    print("All tests passed! 🎉")


if __name__ == "__main__":
    run_all_tests()
