from jax import numpy as jnp
from jax.scipy import sparse


class FinDiffBase:
    def __init__(self, axis, order):
        self.axis = axis
        self.order = order

    def guard_valid_target(self, f):
        try:
            f.shape[self.axis]
        except AttributeError as err:
            raise ValueError(
                "Diff objects can only be applied to arrays or evaluated(!) functions returning arrays"
            ) from err

        if jnp.issubdtype(f.dtype, jnp.integer):
            f = f.astype(jnp.float64)
        return f
    
    def apply_to_array(self, yd, y, weights, off_slices, ref_slice, dim):
        """Applies the finite differences only to slices along a given axis"""

        ndims = len(y.shape)

        all = slice(None, None, 1)

        ref_multi_slice = [all] * ndims
        ref_multi_slice[dim] = ref_slice

        for w, s in zip(weights, off_slices):
            off_multi_slice = [all] * ndims
            off_multi_slice[dim] = s
            if abs(1 - w) < 1.0e-14:
                yd[tuple(ref_multi_slice)] += y[tuple(off_multi_slice)]
            else:
                yd[tuple(ref_multi_slice)] += w * y[tuple(off_multi_slice)]

    
    def shift_slice(self, sl, off, max_index):

        if sl.start + off < 0 or sl.stop + off > max_index:
            raise IndexError("Shift slice out of bounds")

        return slice(sl.start + off, sl.stop + off, sl.step)
    
    def matrix(self, shape):
        siz = jnp.prod(shape)
        mat = 