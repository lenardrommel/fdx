"""Public interface wrappers for the finite-difference expression system."""

import jax
from fdx.grids import make_axis
from fdx.operators import Diff as _Diff, Expression


@jax.tree_util.register_pytree_node_class
class Diff(_Diff):
    """Partial derivative operator with a `findiff`-compatible constructor."""

    def __init__(self, axis, grid=None, periodic=False, acc=_Diff.DEFAULT_ACC):
        """Represents a partial derivative (along one axis).

        For higher derivatives, exponentiate. For mixed partial derivatives, multiply. See
        examples below.

        Examples
        --------
        Set up grid (equidistant here):
            >>> from jax import numpy as jnp
            >>> x = jnp.linspace(0, 10, 100)

        The array to differentiate
            >>> f = jnp.sin(x) # as an example

        Define the first derivative:
            >>> from findiff import Diff
            >>> d_dx = Diff(0)
            >>> d_dx.set_grid({0: x[1] - x[0]})

        Now apply it:
            >>> df_dx = d_dx(f)

        The second derivative is constructed by exponentiation:
            >>> d2_dx2 = d_dx ** 2
            >>> d2f_dx2 = d2_dx2(f)

        In multiple dimensions with meshed grids, the usage is accordingly:
            >>> x = y = z = jnp.linspace(0, 10, 100)
            >>> dx = dy = dz = x[1] - x[0]
            >>> X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            >>> f = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
            >>> d_dx = Diff(0)
            >>> d_dy = Diff(1)
            >>> d_dz = Diff(2)
            >>> d3_dxdydz = d_dx * d_dy * d_dz
            >>> d3_dxdydz.set_grid({0: dx, 1: dy, 2: dz})
            >>> d3f_dxdydz = d3_dxdydz(f)
        """
        grid_axis = make_axis(axis, grid, periodic)
        super().__init__(axis, grid_axis, acc)

    def tree_flatten(self):
        """Flatten into (children, aux_data) for JAX pytree."""
        children = (self._differentiator,) if self._differentiator is not None else ()
        aux_data = (self.dim, self._order, self.acc, self._axis)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from (children, aux_data)."""
        dim, order, acc, axis = aux_data
        obj = object.__new__(cls)
        Expression.__init__(obj)
        obj.dim = dim
        obj.acc = acc
        obj._order = order
        obj._axis = axis
        obj._differentiator = children[0] if children else None
        return obj

