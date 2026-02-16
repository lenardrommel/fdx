"""Finite-difference stencils and stencil application utilities."""

import math
from itertools import product

from jax import numpy as jnp

from fdx.utils import to_index_tuple, to_long_index


class StencilSet:
    """Collection of point stencils for different boundary locations."""

    def __init__(self, diff_op, shape):
        """Create a stencil set for a given operator and field shape.

        Parameters
        ----------
        diff_op
            Operator providing a dense `matrix(shape)` representation.
        shape
            Shape of the discretized field.
        """
        self.shape = shape
        self.diff_op = diff_op
        self.char_pts = self._det_characteristic_points()

        self.data = {}

        self._create_stencil()

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def apply(self, u, idx0):
        """Apply the appropriate point-stencil at a single index.

        Parameters
        ----------
        u
            Array containing the field values.
        idx0
            Grid index at which to evaluate the derivative.

        Returns
        -------
        float
            Derivative value at `idx0`.
        """
        if not hasattr(idx0, "__len__"):
            idx0 = (idx0,)

        typ: list[str] = []
        for axis in range(len(self.shape)):
            if idx0[axis] == 0:
                typ.append("L")
            elif idx0[axis] == self.shape[axis] - 1:
                typ.append("H")
            else:
                typ.append("C")
        typ_tuple = tuple(typ)

        stl = self.data[typ_tuple]
        idx0 = jnp.array(idx0)
        du = 0.0
        for o, c in stl.items():
            idx = idx0 + o
            du += c * u[tuple(idx)]

        return du

    def apply_all(self, u):
        """Apply the stencil to all grid points.

        Parameters
        ----------
        u
            Array containing the field values.

        Returns
        -------
        jax.Array
            Array of derivatives with the same shape as ``u``.
        """
        assert self.shape == u.shape
        return (self._matrix @ u.ravel()).reshape(u.shape)

    def _create_stencil(self):
        matrix = self.diff_op.matrix(self.shape)
        self._matrix = matrix  # cache for apply_all

        for pt in self.char_pts:
            char_point_stencil: dict[tuple[int, ...], float] = {}
            self.data[pt] = char_point_stencil

            index_tuple_for_char_pt = self._typical_index_tuple_for_char_point(pt)
            long_index_for_char_pt = to_long_index(index_tuple_for_char_pt, self.shape)

            row = matrix[long_index_for_char_pt, :]
            long_row_inds, long_col_inds = row.nonzero()

            for long_offset_ind in long_col_inds:
                offset_ind_tuple = jnp.array(
                    to_index_tuple(long_offset_ind, self.shape), dtype=int
                )
                offset_ind_tuple -= jnp.array(index_tuple_for_char_pt, dtype=int)
                char_point_stencil[tuple(offset_ind_tuple)] = float(row[0, long_offset_ind])

    def _typical_index_tuple_for_char_point(self, pt):
        index_tuple_for_char_pt = []
        for axis, key in enumerate(pt):
            if key == "L":
                index_tuple_for_char_pt.append(0)
            elif key == "C":
                index_tuple_for_char_pt.append(self.shape[axis] // 2)
            else:
                index_tuple_for_char_pt.append(self.shape[axis] - 1)
        return tuple(index_tuple_for_char_pt)

    def _det_characteristic_points(self):
        shape = self.shape
        ndim = len(shape)
        typ = [("L", "C", "H")] * ndim
        return product(*typ)


class Stencil:
    """Finite-difference stencil defined by offsets and derivative terms."""

    def __init__(self, offsets, partials, spacings=None):
        """Create a stencil.

        Parameters
        ----------
        offsets
            Offset locations used in the stencil. For 1D, a sequence of ints.
            For higher dimensions, a sequence of index-tuples.
        partials
            Mapping from multi-index derivative powers to weights.
        spacings
            Grid spacings per axis. If a scalar is given, it is broadcast.
        """
        self.partials = partials
        self.max_order = 100
        if not hasattr(offsets[0], "__len__"):
            ndims = 1
            self.offsets = [(off,) for off in offsets]
        else:
            ndims = len(offsets[0])
            self.offsets = offsets

        if spacings is None:
            spacings = [1] * ndims
        elif not hasattr(spacings, "__len__"):
            spacings = [spacings] * ndims
        assert len(spacings) == ndims
        self.spacings = spacings
        self.ndims = ndims
        self.sol, self.sol_as_dict = self._make_stencil()

    def __call__(self, f, at=None, on=None):
        """Apply the stencil on a field.

        Exactly one of `at` and `on` must be provided.

        Parameters
        ----------
        f
            Array containing the field values.
        at
            Single index at which to evaluate the stencil.
        on
            Mask or multi-slice defining where to apply the stencil.

        Returns
        -------
        jax.Array | float
            Stencil evaluation on the specified location(s).
        """
        if at is not None and on is None:
            return self._apply_at_single_point(f, at)
        if at is None and on is not None:
            if isinstance(on[0], slice):
                return self.apply_on_multi_slice(f, on)
            else:
                return self.apply_on_mask(f, on)
        raise Exception("Cannot specify both *at* and *on* parameters.")

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return str(self.values)

    def apply_on_mask(self, f, mask):
        """Apply the stencil to entries selected by a boolean mask.

        Parameters
        ----------
        f
            Array containing the field values.
        mask
            Boolean mask with the same shape as `f`.

        Returns
        -------
        jax.Array
            Array with accumulated stencil evaluations at masked positions.
        """
        result = jnp.zeros_like(f)
        for offset, coeff in self.values.items():
            offset_mask = self._make_offset_mask(mask, offset)
            result = result.at[mask].add(coeff * f[offset_mask])
        return result

    def apply_on_multi_slice(self, f, on):
        """Apply the stencil on a rectangular region defined by slices.

        Parameters
        ----------
        f
            Array containing the field values.
        on
            Sequence of `slice` objects, one per axis.

        Returns
        -------
        jax.Array
            Array with stencil values accumulated on the sliced region.
        """
        result = jnp.zeros_like(f)
        base_mslice = [
            self._canonic_slice(sl, f.shape[axis]) for axis, sl in enumerate(on)
        ]

        for off, coeff in self.values.items():
            off_mslice = list(base_mslice)
            for axis, off_ in enumerate(off):
                start = base_mslice[axis].start + off_
                stop = base_mslice[axis].stop + off_
                off_mslice[axis] = slice(start, stop)

            result = result.at[tuple(base_mslice)].add(coeff * f[tuple(off_mslice)])

        return result

    def _apply_at_single_point(self, f, at):
        result = 0.0
        at = jnp.array(at)
        for off, coeff in self.values.items():
            off = jnp.array(off)
            eval_at = at + off
            if jnp.any(eval_at < 0) or not jnp.all(eval_at < f.shape):
                raise Exception("Cannot evaluate outside of grid.")
            result += coeff * f[tuple(eval_at)]
        return result

    def _make_offset_mask(self, mask, offset):
        offset_mask = jnp.full_like(mask, fill_value=False, dtype=bool)
        mslice_off = []
        mslice_base = []
        for off_ in offset:
            if off_ == 0:
                sl_off = slice(None, None)
                sl_base = slice(None, None)
            elif off_ > 0:
                sl_off = slice(off_, None)
                sl_base = slice(None, -off_)
            else:
                sl_off = slice(None, off_)
                sl_base = slice(-off_, None)
            mslice_off.append(sl_off)
            mslice_base.append(sl_base)

        # Use .at[].set() instead of direct assignment
        offset_mask = offset_mask.at[tuple(mslice_base)].set(mask[tuple(mslice_off)])
        return offset_mask

    def _canonic_slice(self, sl: slice, length: int) -> slice:
        start = sl.start if sl.start is not None else 0
        if start < 0:
            start = length - start
        stop = sl.stop if sl.stop is not None else 0
        if stop < 0:
            stop = length - start
        return slice(start, stop)

    @property
    def values(self):
        """Dictionary mapping offset tuples to stencil coefficients."""
        return self.sol_as_dict

    @property
    def accuracy(self):
        """Estimated accuracy order of the stencil."""
        return self._calc_accuracy()

    def _calc_accuracy(self):
        tol = 1.0e-6
        deriv_order = 0
        for pows in self.partials.keys():
            order = sum(pows)
            if order > deriv_order:
                deriv_order = order
        for order in range(deriv_order, deriv_order + 10):
            terms = self._multinomial_powers(order)
            for term in terms:
                row = self._system_matrix_row(term)
                resid = sum(float(s) * float(r) for s, r in zip(self.sol, row))
                if abs(resid) > tol and term not in self.partials:
                    return order - deriv_order

    def _make_stencil(self):
        sys_matrix, taylor_terms = self._system_matrix()
        rhs = [0] * len(self.offsets)

        for i, term in enumerate(taylor_terms):
            if term in self.partials:
                weight = self.partials[term]
                multiplicity = jnp.prod(jnp.array([math.factorial(a) for a in term]))
                vol = jnp.prod(
                    jnp.array([self.spacings[j] ** term[j] for j in range(self.ndims)])
                )
                rhs[i] = weight * multiplicity / vol

        sol = jnp.linalg.solve(jnp.array(sys_matrix), jnp.array(rhs))
        assert len(sol) == len(self.offsets)
        return sol, {off: coef for off, coef in zip(self.offsets, sol) if coef != 0}

    def _system_matrix(self):
        rows = []
        used_taylor_terms = []
        for order in range(self.max_order):
            taylor_terms = self._multinomial_powers(order)
            for term in taylor_terms:
                rows.append(self._system_matrix_row(term))
                used_taylor_terms.append(term)
                if not self._rows_are_linearly_independent(rows):
                    rows.pop()
                    used_taylor_terms.pop()
                if len(rows) == len(self.offsets):
                    return jnp.array(rows), used_taylor_terms
        raise Exception("Not enough terms. Try to increase max_order.")

    def _system_matrix_row(self, powers):
        row = []
        for a in self.offsets:
            value = 1
            for i, power in enumerate(powers):
                value *= a[i] ** power
            row.append(value)
        return row

    def _multinomial_powers(self, the_sum):
        """Returns all tuples of a given dimension that add up to the_sum."""
        all_combs = list(product(range(the_sum + 1), repeat=self.ndims))
        return [tpl for tpl in all_combs if sum(tpl) == the_sum]

    def _rows_are_linearly_independent(self, matrix):
        """Checks the linear independence of the rows of a matrix."""
        matrix = jnp.array(matrix).astype(jnp.float32)
        return jnp.linalg.matrix_rank(matrix) == len(matrix)
