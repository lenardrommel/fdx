# findiff.py

import jax
from jax import lax
from jax import numpy as jnp
from jax.scipy import sparse

from fdx.coefs import coefficients, coefficients_non_uni
from fdx.config import get_precision
from fdx.grids import EquidistantAxis, GridAxis, NonEquidistantAxis
from fdx.utils import (
    get_long_indices_for_all_grid_points_as_1d_array,
    get_long_indices_for_all_grid_points_as_ndarray,
    to_long_index,
)

jax.config.update("jax_enable_x64", True)


def build_differentiator(order: int, axis: GridAxis, acc):
    if isinstance(axis, EquidistantAxis):
        if not axis.periodic:
            return _FinDiffUniform(axis.dim, order, axis.spacing, acc)
        else:
            return _FinDiffUniformPeriodic(axis.dim, order, axis.spacing, acc)
    elif isinstance(axis, NonEquidistantAxis):
        if not axis.periodic:
            return _FinDiffNonUniform(axis.dim, order, axis.coords, acc)
        else:
            raise NotImplementedError("Periodic nonuniform axes not yet implemented")
    else:
        raise TypeError("Unknown axis type.")


class _FinDiffBase:
    def __init__(self, axis, order, dtype=jnp.float64):
        self.axis = axis
        self.order = order
        self._dtype = dtype

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

    def apply_to_array(self, yd, y, weights, offsets, ref_start, ref_size, dim):
        """Applies finite differences using JAX-compatible dynamic slicing.

        Args:
            yd: output array to accumulate into
            y: input array
            weights: coefficients for finite differences
            offsets: offset indices for each stencil point
            ref_start: starting index of reference region
            ref_size: size of reference region
            dim: axis along which to apply differences
        """
        ndims = len(y.shape)

        for w, offset in zip(weights, offsets):
            start_idx = ref_start + offset

            start_indices = jnp.asarray([0] * ndims)
            start_indices = start_indices.at[dim].set(start_idx)

            slice_sizes = list(y.shape)
            slice_sizes[dim] = ref_size

            y_slice = lax.dynamic_slice(y, start_indices, slice_sizes)

            update_start_indices = jnp.asarray([0] * ndims)
            update_start_indices = update_start_indices.at[dim].set(ref_start)

            ref_slice = lax.dynamic_slice(yd, update_start_indices, slice_sizes)

            updated_slice = ref_slice + w * y_slice

            # Write back using dynamic update
            yd = lax.dynamic_update_slice(yd, updated_slice, update_start_indices)

        return yd

    def shift_slice(self, sl: slice, off: int, max_index: int) -> slice:
        # if sl.start + off < 0 or sl.stop + off > max_index:
        #     raise IndexError("Shift slice out of bounds")
        return slice(sl.start + off, sl.stop + off, sl.step)

    def matrix(self, shape):
        siz = jnp.prod(*shape)
        mat = jnp.zeros((siz, siz))
        mat = self.write_matrix_entries(mat, shape)
        return mat

    def write_matrix_entries(self, mat, shape):
        raise NotImplementedError


class _FinDiffUniform(_FinDiffBase):
    def __init__(self, axis, order, spacing, acc):
        super().__init__(axis, order)
        self.spacing = spacing
        self.acc = acc
        coef_schemes = coefficients(self.order, acc)

        self.forward = coef_schemes["forward"]
        self.backward = coef_schemes["backward"]
        self.center = coef_schemes["center"]

    def __call__(self, f):
        f = self.guard_valid_target(f)
        npts = int(f.shape[self.axis])
        fd = jnp.zeros_like(f)
        num_bndry_points = len(self.center["coefficients"]) // 2

        fd = self._apply_central_coefs(f, fd, npts, num_bndry_points)

        fd = self._apply_forward_coefs(f, fd, npts, num_bndry_points)

        fd = self._apply_backward_coefs(f, fd, npts, num_bndry_points)

        h_inv = 1.0 / self.spacing**self.order
        return fd * h_inv

    def _apply_backward_coefs(self, f, fd, npts, num_bndry_points):
        weights = self.backward["coefficients"]
        offsets = self.backward["offsets"]

        ref_start = npts - num_bndry_points
        ref_size = num_bndry_points

        return self.apply_to_array(
            fd, f, weights, offsets, ref_start, ref_size, self.axis
        )

    def _apply_forward_coefs(self, f, fd, npts, num_bndry_points):
        weights = self.forward["coefficients"]
        offsets = self.forward["offsets"]

        ref_start = 0
        ref_size = num_bndry_points

        return self.apply_to_array(
            fd, f, weights, offsets, ref_start, ref_size, self.axis
        )

    def _apply_central_coefs(self, f, fd, npts, num_bndry_points):
        weights = self.center["coefficients"]
        offsets = self.center["offsets"]

        ref_start = num_bndry_points
        ref_size = npts - 2 * num_bndry_points

        return self.apply_to_array(
            fd, f, weights, offsets, ref_start, ref_size, self.axis
        )

    def write_matrix_entries(self, mat, shape):
        long_indices_nd = get_long_indices_for_all_grid_points_as_ndarray(shape)
        for scheme in ["center", "forward", "backward"]:
            offsets_long = self._convert_1D_offsets_to_long_indices(
                self.axis, getattr(self, scheme)["offsets"], shape
            )

            multi_slice = self._get_multislice_for_scheme(self.axis, scheme, shape)
            Is = long_indices_nd[tuple(multi_slice)].reshape(-1)

            coefs = getattr(self, scheme)["coefficients"]
            for o, c in zip(offsets_long, coefs):
                v = c / self.spacing**self.order
                mat = mat.at[Is, Is + o].set(v)

        return mat

    def _get_multislice_for_scheme(self, axis, scheme, shape):
        ndims = len(shape)
        multi_slice = [slice(None, None)] * ndims
        nside = len(self.center["coefficients"]) // 2
        if scheme == "center":
            multi_slice[axis] = slice(nside, -nside)
        elif scheme == "forward":
            multi_slice[axis] = slice(0, nside)
        else:
            multi_slice[axis] = slice(-nside, None)
        return multi_slice

    def _convert_1D_offsets_to_long_indices(self, axis, offsets_1d, shape):
        ndims = len(shape)
        offsets_long = []
        for o_1d in offsets_1d:
            o_nd = jnp.zeros(ndims, dtype=int)
            o_nd = o_nd.at[axis].set(o_1d)
            o_long = to_long_index(o_nd, shape)
            offsets_long.append(o_long)
        return offsets_long


class _FinDiffUniformPeriodic(_FinDiffBase):
    def __init__(self, axis, order, spacing, acc):
        super().__init__(axis, order)
        self.spacing = spacing
        self.acc = acc
        self.coefs = coefficients(self.order, acc)["center"]

    def __call__(self, f):
        f = self.guard_valid_target(f)

        fd = jnp.zeros_like(f)
        for off, coef in zip(self.coefs["offsets"], self.coefs["coefficients"]):
            fd = fd + coef * jnp.roll(f, -off, axis=self.axis)
        h_inv = 1.0 / self.spacing**self.order
        return fd * h_inv

    def write_matrix_entries(self, mat, shape):
        Is = get_long_indices_for_all_grid_points_as_1d_array(shape)
        h_inv = 1 / self.spacing**self.order
        for o, c in zip(self.coefs["offsets"], self.coefs["coefficients"]):
            Is_off = self._get_offset_indices_long(o, shape)
            mat = mat.at[Is, Is_off].set(c * h_inv)
        return mat

    def _get_offset_indices_long(self, o, shape):
        ndims = len(shape)
        idxs_short = [jnp.arange(n) for n in shape]
        idxs_short[self.axis] = jnp.roll(jnp.arange(shape[self.axis]), -o)
        grid = jnp.meshgrid(*idxs_short, indexing="ij")
        index_tuples = jnp.stack(grid, axis=-1).reshape(-1, ndims)
        Is_off = jnp.ravel_multi_index(index_tuples.T, shape)
        return Is_off


class _FinDiffNonUniform(_FinDiffBase):
    def __init__(self, axis, order, coords, acc):
        super().__init__(axis, order)
        self.coords = coords
        self.acc = acc

    def __call__(self, y):
        """Apply a finite-difference derivative on a non-uniform, non-periodic axis.

        Uses location-dependent coefficients computed from the actual coordinate
        values (no uniform spacing assumption). Boundary points automatically
        use appropriate one-sided stencils.
        """
        y = self.guard_valid_target(y)

        # Move target axis to front for simpler indexing: (n, ...)
        y_m = jnp.moveaxis(y, self.axis, 0)
        n = y_m.shape[0]
        yd_m = jnp.zeros_like(y_m)

        def body(i, acc_arr):
            coef = coefficients_non_uni(self.order, self.acc, self.coords, i)
            ws = coef["coefficients"]  # (k,)
            offs = coef["offsets"].astype(jnp.int32)  # (k,)

            idxs = i + offs  # (k,)
            # Gather the stencil along the moved axis (0)
            stencil_vals = jnp.take(y_m, idxs, axis=0)
            # Weighted sum across stencil dimension
            val = jnp.tensordot(ws, stencil_vals, axes=(0, 0))
            acc_arr = acc_arr.at[i].set(val)
            return acc_arr

        yd_m = lax.fori_loop(0, n, body, yd_m)
        # Move axis back to original place
        return jnp.moveaxis(yd_m, 0, self.axis)

    def write_matrix_entries(self, mat, shape):
        """Optional: assemble a dense operator matrix.

        For non-uniform grids, coefficients vary per row. To avoid a heavy
        dense build in the common path, leave this unimplemented for now.
        """
        raise NotImplementedError("Matrix assembly for non-uniform grids not implemented")
#         multi_slice = [slice(None, None)] * ndims
#         ref_multi_slice = [slice(None, None)] * ndims

#         for i, _ in enumerate(self.coords):
#             coefs = self.coef_list[i]
#             ref_multi_slice[dim] = i

#             for off, w in zip(coefs["offsets"], coefs["coefficients"]):
#                 multi_slice[dim] = i + off
#                 yd[tuple(ref_multi_slice)] += w * y[tuple(multi_slice)]

#         return yd


#     def write_matrix_entries(self, mat, shape):
#         coords = self.coords

#         short_inds = get_list_of_multiindex_tuples(shape)

#         coef_dicts = []
#         for i in range(len(coords)):
#             coef_dicts.append(coefficients_non_uni(self.order, self.acc, coords, i))

#         long_inds = get_long_indices_for_all_grid_points_as_ndarray(shape)
#         for base_ind_long, base_ind_short in enumerate(short_inds):
#             cd = coef_dicts[base_ind_short[self.axis]]
#             cs, os = cd["coefficients"], cd["offsets"]
#             for c, o in zip(cs, os):
#                 off_short = jnp.zeros(len(shape), dtype=int)
#                 off_short[self.axis] = int(o)
#                 off_ind_short = jnp.array(base_ind_short, dtype=int) + off_short
#                 off_long = long_inds[tuple(off_ind_short)]

#                 mat[base_ind_long, off_long] += c
