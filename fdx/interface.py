from fdx.grids import make_axis
from fdx.operators import Diff as _Diff


class Diff(_Diff):
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
