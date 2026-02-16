"""Finite-difference differentiators for uniform and non-uniform grids."""

import math
import equinox as eqx
import jax
from jax import lax
from jax import numpy as jnp

from fdx.coefs import coefficients, coefficients_non_uni, precompute_all_non_uni_coefficients
from fdx.grids import EquidistantAxis, GridAxis, NonEquidistantAxis
from fdx.utils import (
    get_long_indices_for_all_grid_points_as_1d_array,
    get_long_indices_for_all_grid_points_as_ndarray,
    to_long_index,
)

jax.config.update("jax_enable_x64", True)


@jax.custom_jvp
def apply_fd(diff, f):
    """Apply a finite-difference differentiator to an array.

    This function is decorated with ``custom_jvp`` so that JAX's
    automatic differentiation correctly recognises that ``D`` is linear:
    ``jvp(D, (f,), (f_dot,)) = (D(f), D(f_dot))``.
    """
    return diff._apply_impl(f)


@apply_fd.defjvp
def _apply_fd_jvp(primals, tangents):
    diff, f = primals
    _, f_dot = tangents
    primal_out = apply_fd(diff, f)
    tangent_out = apply_fd(diff, f_dot)
    return primal_out, tangent_out


def build_differentiator(order: int, axis: GridAxis, acc):
    """Construct a finite-difference differentiator for a given axis.

    Parameters
    ----------
    order
        Derivative order.
    axis
        Grid axis descriptor.
    acc
        Accuracy order.

    Returns
    -------
    _FinDiffBase
        Differentiator instance appropriate for `axis`.
    """
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


class _FinDiffBase(eqx.Module):
    """Base class for finite-difference differentiators.

    Subclasses are automatically registered as JAX pytrees via ``eqx.Module``.
    """

    axis: int = eqx.field(static=True)
    order: int = eqx.field(static=True)

    def guard_valid_target(self, f):
        """Validate and cast the input array."""
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
        """Apply finite differences using JAX-compatible dynamic slicing."""
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
        """Shift a slice by an offset."""
        return slice(sl.start + off, sl.stop + off, sl.step)

    def matrix(self, shape):
        """Return a dense matrix representation for a given field shape."""
        siz = math.prod(shape)
        mat = jnp.zeros((siz, siz))
        mat = self.write_matrix_entries(mat, shape)
        return mat

    def write_matrix_entries(self, mat, shape):
        """Write operator entries into a dense matrix."""
        raise NotImplementedError


class _FinDiffUniform(_FinDiffBase):
    """Finite-difference differentiator for uniform, non-periodic grids."""

    spacing: jax.Array
    acc: int = eqx.field(static=True)
    forward: dict
    backward: dict
    center: dict

    def __init__(self, axis, order, spacing, acc):
        self.axis = axis
        self.order = order
        self.spacing = spacing
        self.acc = acc
        coef_schemes = coefficients(self.order, acc)
        self.forward = coef_schemes["forward"]
        self.backward = coef_schemes["backward"]
        self.center = coef_schemes["center"]

    def __call__(self, f):
        """Apply the derivative to an input array."""
        f = self.guard_valid_target(f)
        return apply_fd(self, f)

    def _apply_impl(self, f):
        """Core implementation: convolution for central, dynamic slicing for boundaries."""
        npts = int(f.shape[self.axis])
        fd = jnp.zeros_like(f)
        num_bndry_points = len(self.center["coefficients"]) // 2

        # Central region: use JAX convolution (single fused XLA op)
        fd = self._apply_central_conv(f, fd, npts, num_bndry_points)

        # Boundary regions: keep apply_to_array (only a few points)
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

    def _apply_central_conv(self, f, fd, npts, num_bndry_points):
        """Apply central FD stencil via JAX 1D convolution (single XLA op)."""
        weights = jnp.array(self.center["coefficients"], dtype=f.dtype)
        k = len(weights)
        dim = self.axis
        ndims = f.ndim

        # Reshape f so the target axis is the last spatial dim for conv:
        # Move target axis to position -1, then add batch+channel dims for lax.conv
        perm = [i for i in range(ndims) if i != dim] + [dim]
        inv_perm = [0] * ndims
        for new_pos, old_pos in enumerate(perm):
            inv_perm[old_pos] = new_pos

        f_t = jnp.transpose(f, perm)  # (..., n_axis)
        orig_shape = f_t.shape

        # Flatten all dims except the last into a batch dim: (B, n_axis)
        spatial_n = orig_shape[-1]
        batch_size = math.prod(orig_shape[:-1])
        f_flat = f_t.reshape(batch_size, spatial_n)

        # lax.conv_general_dilated expects (batch, channels, spatial...)
        # Use depthwise-like 1D conv: treat batch as channels, convolve each independently
        # Shape: (batch, 1, spatial) with kernel (1, 1, k)
        f_conv_in = f_flat[:, None, :]  # (B, 1, N)

        # lax.conv_general_dilated does cross-correlation, so use coefficients directly
        kernel = weights.reshape(1, 1, k)  # (out_ch, in_ch, k)

        # Valid convolution (no padding) gives output size = N - k + 1
        conv_out = lax.conv_general_dilated(
            f_conv_in.astype(weights.dtype),
            kernel,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )  # (B, 1, N - k + 1)

        conv_result = conv_out[:, 0, :]  # (B, N - k + 1)

        # Write conv result into the central region of fd
        ref_start = num_bndry_points
        ref_size = npts - 2 * num_bndry_points

        fd_t = jnp.transpose(fd, perm)
        fd_flat = fd_t.reshape(batch_size, spatial_n)
        fd_flat = fd_flat.at[:, ref_start : ref_start + ref_size].set(
            conv_result.astype(fd.dtype)
        )

        fd_t = fd_flat.reshape(orig_shape)
        fd = jnp.transpose(fd_t, inv_perm)
        return fd

    def write_matrix_entries(self, mat, shape):
        """Assemble dense matrix entries using batched COO scatter."""
        long_indices_nd = get_long_indices_for_all_grid_points_as_ndarray(shape)
        h_inv = 1.0 / self.spacing**self.order

        all_rows = []
        all_cols = []
        all_vals = []

        for scheme in ["center", "forward", "backward"]:
            offsets_long = self._convert_1D_offsets_to_long_indices(
                self.axis, getattr(self, scheme)["offsets"], shape
            )
            multi_slice = self._get_multislice_for_scheme(self.axis, scheme, shape)
            Is = long_indices_nd[tuple(multi_slice)].reshape(-1)
            coefs = getattr(self, scheme)["coefficients"]

            for o, c in zip(offsets_long, coefs):
                all_rows.append(Is)
                all_cols.append(Is + o)
                all_vals.append(jnp.full_like(Is, c * h_inv, dtype=mat.dtype))

        rows = jnp.concatenate(all_rows)
        cols = jnp.concatenate(all_cols)
        vals = jnp.concatenate(all_vals)
        return mat.at[rows, cols].add(vals)

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
    """Finite-difference differentiator for uniform, periodic grids."""

    spacing: jax.Array
    acc: int = eqx.field(static=True)
    coefs: dict

    def __init__(self, axis, order, spacing, acc):
        self.axis = axis
        self.order = order
        self.spacing = spacing
        self.acc = acc
        self.coefs = coefficients(self.order, acc)["center"]

    def __call__(self, f):
        """Apply the derivative to an input array."""
        f = self.guard_valid_target(f)
        return apply_fd(self, f)

    def _apply_impl(self, f):
        """Core implementation: circular pad + convolution."""
        weights = jnp.array(self.coefs["coefficients"], dtype=f.dtype)
        k = len(weights)
        half = k // 2
        dim = self.axis
        ndims = f.ndim

        # Move target axis to last, flatten rest into batch
        perm = [i for i in range(ndims) if i != dim] + [dim]
        inv_perm = [0] * ndims
        for new_pos, old_pos in enumerate(perm):
            inv_perm[old_pos] = new_pos

        f_t = jnp.transpose(f, perm)
        orig_shape = f_t.shape
        spatial_n = orig_shape[-1]
        batch_size = math.prod(orig_shape[:-1])
        f_flat = f_t.reshape(batch_size, spatial_n)

        # Circular pad along spatial axis
        f_padded = jnp.concatenate(
            [f_flat[:, -half:], f_flat, f_flat[:, :half]], axis=-1
        )

        # lax.conv_general_dilated does cross-correlation, use coefficients directly
        f_conv_in = f_padded[:, None, :]  # (B, 1, N + 2*half)
        kernel = weights.reshape(1, 1, k)

        conv_out = lax.conv_general_dilated(
            f_conv_in.astype(weights.dtype),
            kernel,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )  # (B, 1, N)

        fd_flat = conv_out[:, 0, :]  # (B, N)
        fd_t = fd_flat.reshape(orig_shape)
        fd = jnp.transpose(fd_t, inv_perm)

        h_inv = 1.0 / self.spacing**self.order
        return (fd * h_inv).astype(f.dtype)

    def write_matrix_entries(self, mat, shape):
        """Assemble dense matrix entries using batched COO scatter."""
        Is = get_long_indices_for_all_grid_points_as_1d_array(shape)
        h_inv = 1 / self.spacing**self.order

        all_rows = []
        all_cols = []
        all_vals = []

        for o, c in zip(self.coefs["offsets"], self.coefs["coefficients"]):
            Is_off = self._get_offset_indices_long(o, shape)
            all_rows.append(Is)
            all_cols.append(Is_off)
            all_vals.append(jnp.full_like(Is, c * h_inv, dtype=mat.dtype))

        rows = jnp.concatenate(all_rows)
        cols = jnp.concatenate(all_cols)
        vals = jnp.concatenate(all_vals)
        return mat.at[rows, cols].add(vals)

    def _get_offset_indices_long(self, o, shape):
        ndims = len(shape)
        idxs_short = [jnp.arange(n) for n in shape]
        idxs_short[self.axis] = jnp.roll(jnp.arange(shape[self.axis]), -o)
        grid = jnp.meshgrid(*idxs_short, indexing="ij")
        index_tuples = jnp.stack(grid, axis=-1).reshape(-1, ndims)
        Is_off = jnp.ravel_multi_index(index_tuples.T, shape)
        return Is_off


class _FinDiffNonUniform(_FinDiffBase):
    """Finite-difference differentiator for non-uniform, non-periodic grids."""

    acc: int = eqx.field(static=True)
    coords: jnp.ndarray
    _all_coefficients: jnp.ndarray
    _all_offsets: jnp.ndarray

    def __init__(self, axis, order, coords, acc):
        self.axis = axis
        self.order = order
        self.coords = coords
        self.acc = acc

        # Precompute all per-point coefficients as padded arrays (runs once, not traced)
        precomputed = precompute_all_non_uni_coefficients(order, acc, coords)
        self._all_coefficients = precomputed["all_coefficients"]  # (n, max_width)
        self._all_offsets = precomputed["all_offsets"]  # (n, max_width)

    def __call__(self, y):
        """Apply a finite-difference derivative on a non-uniform, non-periodic axis."""
        y = self.guard_valid_target(y)
        return apply_fd(self, y)

    def _apply_impl(self, y):
        """Core implementation using precomputed coefficients.

        The loop body is pure array operations with fixed shapes,
        making it fully traceable under ``jax.jit``.
        """

        # Move target axis to front for simpler indexing: (n, ...)
        y_m = jnp.moveaxis(y, self.axis, 0)
        n = y_m.shape[0]
        yd_m = jnp.zeros_like(y_m)

        all_ws = self._all_coefficients  # (n, max_width)
        all_offs = self._all_offsets  # (n, max_width)

        def body(i, acc_arr):
            ws = all_ws[i]  # (max_width,)
            offs = all_offs[i]  # (max_width,)

            idxs = i + offs  # (max_width,)
            # Gather the stencil along the moved axis (0)
            stencil_vals = jnp.take(y_m, idxs, axis=0)  # (max_width, ...)
            # Weighted sum across stencil dimension (zero-padded entries contribute 0)
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
