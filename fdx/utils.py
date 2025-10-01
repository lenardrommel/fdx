import itertools
from typing import List, Sequence, Tuple

import jax.numpy as jnp


def to_long_index(idx: Sequence[int], shape: Sequence[int]) -> int:
    ndims = len(shape)
    long_idx = 0
    siz = 1
    for axis in range(ndims):
        idx_ = idx[ndims - 1 - axis]
        long_idx += idx_ * siz
        siz *= shape[ndims - 1 - axis]
    return long_idx


def to_index_tuple(long_idx: int, shape: Sequence[int]) -> Tuple[int, ...]:
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
    return get_long_indices_for_all_grid_points_as_1d_array(shape).reshape(shape)


def get_long_indices_for_all_grid_points_as_1d_array(
    shape: Sequence[int],
) -> jnp.ndarray:
    return jnp.arange(jnp.prod(*shape), dtype=jnp.int64)


def get_list_of_multiindex_tuples(shape: Sequence[int]) -> List[Tuple[int, ...]]:
    short_inds = [jnp.arange(shape[k]) for k in range(len(shape))]
    short_inds = list(itertools.product(*short_inds))
    return short_inds
