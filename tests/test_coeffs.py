import pytest
import jax
from jax import numpy as jnp
from fdx.coefs import * 
from findiff.coefs import coefficients as coefficients_np
from findiff.coefs import compute_inverse_Vandermonde as compute_inverse_Vandermonde_np
from findiff.coefs import calc_coefs
from fdx.coefs import compute_coeffs, coefficients, compute_inverse_Vandermonde


def test_compute_inverse_Vandermonde():
    
    c = compute_inverse_Vandermonde(column=2, offsets=[-1, 0, 1])
    c_np = compute_inverse_Vandermonde_np(column=2, offsets=[-1, 0, 1], symbolic=False)
    assert jnp.allclose(c, c_np)


def test_order2_acc2(analytic_inv):
    for analytic_inv in [False, True]:
        c = coefficients(deriv=2, acc=2, analytic_inv=analytic_inv)
        coefs = c["center"]["coefficients"]
        coefs_np = coefficients_np(deriv=2, acc=2, analytic_inv=analytic_inv)["center"]["coefficients"]

        assert jnp.allclose(coefs, coefs_np)


test_order2_acc2(analytic_inv=False)
test_compute_inverse_Vandermonde()

print("All tests passed!")