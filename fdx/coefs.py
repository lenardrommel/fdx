# coefs.py

import math
from itertools import combinations
from typing import Any, Dict, List, Optional

from jax import lax
from jax import numpy as jnp

# import numpy as jnp
from fdx.config import _dtype


def coefficients(
    deriv: int,
    acc: Optional[int] = None,
    offsets: Optional[List[int]] = None,
    symbolic: bool = False,
    analytic_inv: bool = False,
) -> Dict[str, Any]:
    _validate_deriv(deriv)
    if acc is not None and offsets:
        raise ValueError("acc and offsets cannot both be given")

    if offsets:
        if deriv >= len(offsets):
            raise ValueError(
                f"can not compute derivative of order {deriv} using {len(offsets)} offsets."
            )
        return compute_coeffs(deriv, offsets, analytic_inv)

    if acc is None:
        raise ValueError("either acc or offsets has to be given")

    _validate_acc(acc)
    ret = {}
    num_central_coefs = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side_coefs = num_central_coefs // 2

    # Determine central coefficients
    offsets = list(range(-num_side_coefs, num_side_coefs + 1))
    ret["center"] = compute_coeffs(deriv, offsets, analytic_inv)

    # Determine forward coefficients

    if deriv % 2 == 0:
        num_coef = num_central_coefs + 1
    else:
        num_coef = num_central_coefs

    offsets = list(range(num_coef))
    ret["forward"] = compute_coeffs(deriv, offsets, analytic_inv)

    # Determine backward coefficients

    offsets = list(range(-num_coef + 1, 1))
    ret["backward"] = compute_coeffs(deriv, offsets, analytic_inv)

    return ret


def coefficients_non_uni(
    deriv: int, acc: int, coords: jnp.ndarray, idx: int
) -> Dict[str, jnp.ndarray]:
    """
    Calculates the finite difference coefficients for given derivative order and accuracy order.
    Assumes that the underlying grid is non-uniform.

    :param deriv: int > 0: The derivative order.

    :param acc:  even int > 0: The accuracy order.

    :param coords:  1D numpy.ndarray: the coordinates of the axis for the partial derivative

    :param idx:  int: index of the grid position where to calculate the coefficients

    :return: dict with the finite difference coefficients and corresponding offsets
    """

    _validate_deriv(deriv)
    _validate_acc(acc)

    num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
    num_side = num_central // 2

    if deriv % 2 == 0:
        num_coef = num_central + 1
    else:
        num_coef = num_central

    if idx < num_side:
        matrix = _build_matrix_non_uniform(0, num_coef - 1, coords, idx)

        offsets = list(range(num_coef))
        rhs = _build_rhs(offsets, deriv)

        ret = {
            "coefficients": jnp.linalg.solve(matrix, rhs),
            "offsets": jnp.array(offsets),
        }

    elif idx >= len(coords) - num_side:
        matrix = _build_matrix_non_uniform(num_coef - 1, 0, coords, idx)

        offsets = list(range(-num_coef + 1, 1))
        rhs = _build_rhs(offsets, deriv)

        ret = {
            "coefficients": jnp.linalg.solve(matrix, rhs),
            "offsets": jnp.array(offsets),
        }

    else:
        matrix = _build_matrix_non_uniform(num_side, num_side, coords, idx)

        offsets = list(range(-num_side, num_side + 1))
        rhs = _build_rhs(offsets, deriv)

        ret = {
            "coefficients": jnp.linalg.solve(matrix, rhs),
            "offsets": jnp.array([p for p in range(-num_side, num_side + 1)]),
        }

    return ret


def compute_coeffs(
    deriv: int, offsets: List[int], analytic_inv: bool = False
) -> Dict[str, Any]:
    if analytic_inv:
        coefs = compute_inverse_Vandermonde(deriv, offsets)
    else:
        mat = _build_matrix(offsets)
        rhs = _build_rhs(offsets, deriv)
        coefs = jnp.linalg.solve(mat, rhs)

    acc = _calc_accuracy(offsets, coefs, deriv)
    offsets = jnp.array(offsets, dtype=jnp.int32)
    return {"coefficients": coefs, "offsets": offsets, "accuracy": acc}


def _build_matrix_non_uniform(
    p: int, q: int, coords: jnp.ndarray, k: int, dtype: Any = _dtype
) -> jnp.ndarray:
    """Constructs the equation matrix for the finite difference coefficients of non-uniform grids at location k"""
    j_indices = jnp.arange(-p, q + 1)

    powers = jnp.arange(p + q + 1).reshape(-1, 1)

    coord_indices = k + j_indices
    coord_values = coords[coord_indices]
    diffs = coord_values - coords[k]

    A = jnp.power(diffs, powers)

    return A.astype(dtype)


def compute_inverse_Vandermonde(
    column: int, offsets: List[int], dtype: Any = _dtype
) -> jnp.ndarray:
    take = lambda arr, ids: arr[jnp.array(ids)]  # noqa: E731
    prod = jnp.prod
    offsets = jnp.array(offsets, dtype=dtype)

    def minus(x: jnp.ndarray, arr: jnp.ndarray) -> jnp.ndarray:
        return x - arr

    n = len(offsets)
    k = column + 1
    inv_vandermonde_column = []
    if k == n:
        # If the number of offsets matches the derivative order + 1, there is a special
        # case, compare the lower part of the bracket in the equation in proofwiki.
        for j in range(n):
            denom = prod(minus(offsets[j], offsets[:j])) * prod(
                minus(offsets[j], offsets[j + 1 :])
            )
            inv_vandermonde_column.append(1 / denom)
    else:
        # This is the "regular" part of the bracket. First compute the sign that is the
        # same for all entries in the column that we compute
        sign = (-1) ** (n - k)
        for j in range(n):
            # All indices except j
            range_wo_j = list(range(j)) + list(range(j + 1, n))
            # Get all combinations of n-k indices that are ascending and do not contain j
            index_set = combinations(range_wo_j, r=n - k)
            enumerator = sum(prod(take(offsets, list(m))) for m in index_set)
            denominator = prod(minus(offsets[j], take(offsets, range_wo_j)))
            inv_vandermonde_column.append(sign * enumerator / denominator)

    return jnp.array(inv_vandermonde_column, dtype=dtype) * math.factorial(column)


def _build_matrix(offsets: List[int], dtype: Any = _dtype) -> jnp.ndarray:
    """Constructs the equation system matrix for the finite difference coefficients"""
    return jnp.vander(jnp.array(offsets), len(offsets), increasing=True).T.astype(dtype)


def _build_rhs(offsets: List[int], deriv: int, dtype: Any = _dtype) -> jnp.ndarray:
    """The right hand side of the equation system matrix"""
    b = jnp.zeros(len(offsets), dtype=dtype)
    b = b.at[deriv].set(math.factorial(deriv))
    return b


def _calc_accuracy(
    offsets: List[int], coefs: List[float], deriv: int, dtype: Any = _dtype
) -> int:
    """Calculate accuracy using JAX-friendly operations"""
    offsets = jnp.asarray(offsets)
    coefs = jnp.asarray(coefs)

    def cond_fun(state: tuple[int, bool]) -> jnp.ndarray:
        n, found = state
        return jnp.logical_and(~found, n <= 999)

    def body_fun(state: tuple[int, bool]) -> tuple[int, bool]:
        n, _ = state
        powers = jnp.power(offsets, n)
        b = jnp.sum(coefs * powers)
        found = jnp.abs(b) > 1.0e-6
        return (n + 1, found)

    init_state = (deriv + 1, False)
    final_n, found = lax.while_loop(cond_fun, body_fun, init_state)

    accuracy = lax.cond(
        found,
        lambda n: n - deriv - 1,
        lambda n: -1,
        final_n,
    )

    return round(jnp.array(accuracy, dtype=dtype))


def _validate_acc(acc: int) -> None:
    if acc % 2 == 1 or acc <= 0:
        raise ValueError(
            f"Accuracy order acc must be positive EVEN integer. Got {acc}, expected {acc + 1} or {acc - 1}."
        )


def _validate_deriv(deriv: int) -> None:
    if deriv < 0:
        raise ValueError("Derive degree must be positive integer")
