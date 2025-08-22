import jax
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


def build_differentiator(order: int, axis: GridAxis, acc):
    if isinstance(axis, EquidistantAxis):
        if not axis.periodic:
            return _FinDiffUniform(axis.dim, order, axis.spacing, acc)
        else:
            return _FinDiffUniformPeriodic(axis.dim, order, axis.spacing, acc)
    elif isinstance(axis, NonEquidistantAxis):
        if not axis.periodic:
            raise NotImplementedError(
                "Non-uniform finite differences for non-periodic axes are not yet implemented."
            )
            # return _FinDiffNonUniform(axis.dim, order, axis.coords, acc)
        else:
            raise NotImplementedError("Periodic nonuniform axes not yet implemented")
    else:
        raise TypeError("Unknown axis type.")


class _FinDiffBase:
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

    def apply_convolution_along_axis(self, f, coeffs, offsets=None):
        if f.ndim == 1:
            return jnp.convolve(f, coeffs[::-1], mode="valid")
        f_moved = jnp.moveaxis(f, self.axis, 0)
        original_shape = f_moved.shape
        f_2d = f_moved.reshape(original_shape[0], -1)

        def conv_1d_column(col):
            return jnp.convolve(col, coeffs[::-1], mode="valid")

        result_2d = jax.vmap(conv_1d_column, in_axes=1, out_axes=1)(f_2d)

        conv_size_reduction = len(coeffs) - 1
        new_axis_size = original_shape[0] - conv_size_reduction
        new_shape = (new_axis_size,) + original_shape[1:]
        result = result_2d.reshape(new_shape)

        # Move axis back to original position
        return jnp.moveaxis(result, 0, self.axis)

    def apply_padded_convolution_along_axis(
        self, f, coeffs, pad_mode="constant", pad_value=0
    ):
        """
        NEW: Alternative that maintains original array size using padding.
        Useful when you want output same size as input.
        """
        pad_width = len(coeffs) // 2

        if f.ndim == 1:
            f_padded = jnp.pad(f, pad_width, mode=pad_mode, constant_values=pad_value)
            return jnp.convolve(f_padded, coeffs[::-1], mode="valid")

        # For multi-dimensional arrays
        pad_spec = [(0, 0)] * f.ndim
        pad_spec[self.axis] = (pad_width, pad_width)

        f_padded = jnp.pad(f, pad_spec, mode=pad_mode, constant_values=pad_value)
        return self.apply_convolution_along_axis(f_padded, coeffs)

    def apply_to_array(self, yd, y, weights, off_slices, ref_slice, dim):
        """Apply FD slices along axis; branch-free, dtype-safe."""
        ndims = y.ndim
        all_ = slice(None)
        ref_multi = [all_] * ndims
        ref_multi[dim] = ref_slice
        ref_tup = tuple(ref_multi)

        # Ensure weights live on the same device/dtype as y
        wv = jnp.asarray(weights, dtype=y.dtype)

        for w, s in zip(wv, off_slices):
            off_multi = [all_] * ndims
            off_multi[dim] = s
            off_tup = tuple(off_multi)
            yd = yd.at[ref_tup].add(w * y[off_tup])
        return yd

    def shift_slice(self, sl, off, max_index):
        if sl.start + off < 0 or sl.stop + off > max_index:
            raise IndexError("Shift slice out of bounds")

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

    def convolve(self, yd, y, num_bndry_points):
        central_part = jnp.convolve(y, self.center["coefficients"][::-1], mode="valid")
        result = yd.at[num_bndry_points:-num_bndry_points].set(central_part)
        length = y.shape[self.axis]

        def left_side_conv(coeffs):
            return jnp.convolve(y[: num_bndry_points * 2], coeffs[::-1], mode="valid")

        def right_side_conv(coeffs):
            slice_start = length - len(coeffs)
            right_slice = y[slice_start:]
            return jnp.convolve(right_slice, coeffs[::-1], mode="valid")

        # Process left boundary
        for i in range(num_bndry_points):
            conv_result = left_side_conv(self.backward["coefficients"][i])
            result = result.at[i].set(conv_result[i])

        # Process right boundary
        for i in range(num_bndry_points):
            conv_result = right_side_conv(self.forward["coefficients"][i])
            result = result.at[length - num_bndry_points + i].set(conv_result[0])

        return result

    def __call__(self, f):
        f = self.guard_valid_target(f)
        npts = f.shape[self.axis]
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
        ref_slice = slice(npts - num_bndry_points, npts, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]
        return self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)

    def _apply_forward_coefs(self, f, fd, npts, num_bndry_points):
        weights = self.forward["coefficients"]
        offsets = self.forward["offsets"]
        ref_slice = slice(0, num_bndry_points, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]
        return self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)

    def _apply_central_coefs(self, f, fd, npts, num_bndry_points):
        weights = self.center["coefficients"]
        offsets = self.center["offsets"]
        ref_slice = slice(num_bndry_points, npts - num_bndry_points, 1)
        off_slices = [
            self.shift_slice(ref_slice, offsets[k], npts) for k in range(len(offsets))
        ]
        return self.apply_to_array(fd, f, weights, off_slices, ref_slice, self.axis)
        # central_part = jnp.convolve(f, weights[::-1], mode="valid")
        # fd = fd.at[ref_slice].set(central_part)
        # return fd

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


# class _FinDiffNonUniform(_FinDiffBase):
#     def __init__(self, axis, order, coords, acc):
#         super().__init__(axis, order)
#         self.coords = coords
#         self.acc = acc
#         self.coef_list = [
#             coefficients_non_uni(order, self.acc, self.coords, i)
#             for i in range(len(coords))
#         ]


#     def __call__(self, y):
#         """The core function to take a partial derivative on a non-uniform grid"""
#         y = self.guard_valid_target(y)

#         dim = self.axis

#         yd = jnp.zeros_like(y)

#         ndims = len(y.shape)
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
