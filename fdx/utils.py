import itertools
from itertools import product

from jax import numpy as jnp


def to_long_index(idx, shape):

    ndims = len(shape)
    long_idx = 0
    siz = 1
    for axis in range(ndims):
        idx_ = idx[ndims - 1 - axis]
        long_idx += idx_ * siz
        siz *= shape[ndims - 1 - axis]

    return long_idx


def get_long_indices_for_all_grid_points_as_ndarray(shape):
    return get_long_indices_for_all_grid_points_as_1d_array(shape).reshape(shape)


def get_long_indices_for_all_grid_points_as_1d_array(shape):
    return jnp.arange(jnp.prod(shape), dtype=jnp.int64)