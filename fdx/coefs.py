"""Finite-difference coefficient generation utilities."""

import math
from itertools import combinations
from typing import Any, Dict, List, Optional

from jax import lax
from jax import numpy as jnp

# import numpy as jnp
from fdx.config import _dtype
from fdx.types import Array


def coefficients(
    deriv: int,
    acc: Optional[int] = None,
    offsets: Optional[List[int]] = None,
    symbolic: bool = False,
    analytic_inv: bool = False,
) -> Dict[str, Any]:
    """Compute finite-difference coefficients for a derivative order.

    Exactly one of `acc` and `offsets` must be provided.

    Parameters
    ----------
    deriv
        Derivative order.
    acc
        Accuracy order (positive even integer) for automatically chosen stencils.
    offsets
        Explicit stencil offsets. If provided, coefficients are computed only for
        these offsets.
    symbolic
        Unused compatibility argument.
    analytic_inv
        Whether to compute coefficients via an analytic inverse Vandermonde
        formula.

    Returns
    -------
    dict
        Dictionary containing coefficient schemes. For automatic stencils this
        includes `"center"`, `"forward"`, and `"backward"`.
    """
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
    deriv: int, acc: int, coords: Array, idx: int
) -> Dict[str, Array]:
    """Compute finite-difference coefficients on a non-uniform 1D grid.

    Parameters
    ----------
    deriv
        Derivative order.
    acc
        Accuracy order (positive even integer).
    coords
        1D coordinate array defining the non-uniform grid.
    idx
        Index of the grid location at which to compute the local stencil.

    Returns
    -------
    dict
        Dictionary with keys `"coefficients"` and `"offsets"`.
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


def precompute_all_non_uni_coefficients(
    deriv: int, acc: int, coords: Array
) -> Dict[str, Array]:
    """Precompute finite-difference coefficients for all non-uniform grid points.

    Returns padded arrays of uniform shape so they can be used inside
    ``lax.fori_loop`` without data-dependent branching.

    Parameters
    ----------
    deriv
        Derivative order.
    acc
        Accuracy order (positive even integer).
    coords
        1D coordinate array defining the non-uniform grid.

    Returns
    -------
    dict
        Dictionary with keys ``"all_coefficients"`` (shape ``(n, max_width)``)
        and ``"all_offsets"`` (shape ``(n, max_width)``), zero-padded.
    """
    _validate_deriv(deriv)
    _validate_acc(acc)

    n = len(coords)
    # Compute per-point coefficients (Python loop — runs once at init, not traced)
    coef_list = []
    for i in range(n):
        coef_list.append(coefficients_non_uni(deriv, acc, coords, i))

    # Determine max stencil width for padding
    max_width = max(len(c["coefficients"]) for c in coef_list)

    # Pad into uniform arrays
    all_coefficients = jnp.zeros((n, max_width))
    all_offsets = jnp.zeros((n, max_width), dtype=jnp.int32)

    for i, c in enumerate(coef_list):
        w = len(c["coefficients"])
        all_coefficients = all_coefficients.at[i, :w].set(c["coefficients"])
        all_offsets = all_offsets.at[i, :w].set(c["offsets"].astype(jnp.int32))

    return {"all_coefficients": all_coefficients, "all_offsets": all_offsets}


def compute_coeffs(
    deriv: int, offsets: List[int], analytic_inv: bool = False
) -> Dict[str, Any]:
    """Compute coefficients for a fixed set of stencil offsets.

    Parameters
    ----------
    deriv
        Derivative order.
    offsets
        Stencil offsets.
    analytic_inv
        Whether to compute coefficients via an analytic inverse Vandermonde
        formula.

    Returns
    -------
    dict
        Dictionary with keys `"coefficients"`, `"offsets"`, and `"accuracy"`.
    """
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
    p: int, q: int, coords: Array, k: int, dtype: Any = _dtype
) -> Array:
    """Constructs the equation matrix for the finite difference coefficients of non-uniform grids at location k."""
    j_indices = jnp.arange(-p, q + 1)

    powers = jnp.arange(p + q + 1).reshape(-1, 1)

    coord_indices = k + j_indices
    coord_values = coords[coord_indices]
    diffs = coord_values - coords[k]

    A = jnp.power(diffs, powers)

    return A.astype(dtype)


def compute_inverse_Vandermonde(
    column: int, offsets: List[int], dtype: Any = _dtype
) -> Array:
    """Compute one column of the inverse Vandermonde system.

    Parameters
    ----------
    column
        Column index (corresponds to derivative order).
    offsets
        Stencil offsets.
    dtype
        Output dtype.

    Returns
    -------
    jax.Array
        Coefficients for the requested column.
    """
    take = lambda arr, ids: arr[jnp.array(ids)]  # noqa: E731
    prod = jnp.prod
    offsets = jnp.array(offsets, dtype=dtype)

    def minus(x: Array, arr: Array) -> Array:
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


def _build_matrix(offsets: List[int], dtype: Any = _dtype) -> Array:
    """Constructs the equation system matrix for the finite difference coefficients."""
    return jnp.vander(jnp.array(offsets), len(offsets), increasing=True).T.astype(dtype)


def _build_rhs(offsets: List[int], deriv: int, dtype: Any = _dtype) -> Array:
    """The right hand side of the equation system matrix."""
    b = jnp.zeros(len(offsets), dtype=dtype)
    b = b.at[deriv].set(math.factorial(deriv))
    return b


def _calc_accuracy(
    offsets: List[int], coefs: List[float], deriv: int, dtype: Any = _dtype
) -> int:
    """Calculate accuracy using JAX-friendly operations."""
    offsets = jnp.asarray(offsets)
    coefs = jnp.asarray(coefs)

    def cond_fun(state: tuple[int, bool]) -> Array:
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
