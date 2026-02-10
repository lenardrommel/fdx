"""Index conversion helpers for reshaped finite-difference operators."""

import itertools
from typing import List, Sequence, Tuple

import jax.numpy as jnp


def to_long_index(idx: Sequence[int], shape: Sequence[int]) -> int:
    """Convert an N-D index to a flattened (row-major) index.

    Parameters
    ----------
    idx
        N-D index tuple.
    shape
        N-D array shape.

    Returns
    -------
    int
        Flattened index corresponding to `idx`.
    """
    ndims = len(shape)
    long_idx = 0
    siz = 1
    for axis in range(ndims):
        idx_ = idx[ndims - 1 - axis]
        long_idx += idx_ * siz
        siz *= shape[ndims - 1 - axis]
    return long_idx


def to_index_tuple(long_idx: int, shape: Sequence[int]) -> Tuple[int, ...]:
    """Convert a flattened (row-major) index to an N-D index tuple.

    Parameters
    ----------
    long_idx
        Flattened index.
    shape
        N-D array shape.

    Returns
    -------
    tuple[int, ...]
        N-D index tuple corresponding to `long_idx`.
    """
    ndims = len(shape)
    idx = jnp.zeros(ndims)
    for k in range(ndims):
        s = jnp.prod(shape[k + 1 :])
        idx = idx.at[k].set(long_idx // s)
        long_idx = long_idx - s * idx[k]
    return tuple(idx)


def get_long_indices_for_all_grid_points_as_ndarray(
    shape: Sequence[int],
) -> jnp.ndarray:
    """Return flattened indices for all grid points as an N-D array.

    Parameters
    ----------
    shape
        Shape of the grid.

    Returns
    -------
    jax.Array
        Array of shape `shape` whose entries are the flattened indices.
    """
    return get_long_indices_for_all_grid_points_as_1d_array(shape).reshape(shape)


def get_long_indices_for_all_grid_points_as_1d_array(
    shape: Sequence[int],
) -> jnp.ndarray:
    """Return flattened indices for all grid points as a 1D array.

    Parameters
    ----------
    shape
        Shape of the grid.

    Returns
    -------
    jax.Array
        1D array of length `prod(shape)` with flattened indices.
    """
    return jnp.arange(jnp.prod(jnp.array(shape)), dtype=jnp.int64)


def get_list_of_multiindex_tuples(shape: Sequence[int]) -> List[Tuple[int, ...]]:
    """Return all N-D index tuples for a given shape.

    Parameters
    ----------
    shape
        Shape of the grid.

    Returns
    -------
    list[tuple[int, ...]]
        List of all index tuples in lexicographic order.
    """
    short_inds = [jnp.arange(shape[k]) for k in range(len(shape))]
    short_inds = list(itertools.product(*short_inds))
    return short_inds
