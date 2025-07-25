import numbers

from jax import numpy as jnp
from linox import LinearOperator
from linox._arithmetic import lmatmul, lmul
from linox._matrix import Diagonal

from fdx.config import _dtype
from fdx.findiff import build_differentiator
from fdx.grids import GridAxis, make_grid


class FieldOperator(LinearOperator):
    def __init__(self, value, shape):
        super().__init__(shape=shape, dtype=_dtype)
        self.value = value

    def __call__(self, f, *args, **kwargs):
        if isinstance(f, (numbers.Number, jnp.ndarray)):
            return self.value * f
        return self.value * super().__call__(f, *args, **kwargs)

    def matrix(self, shape):
        if isinstance(self.value, jnp.ndarray):
            diag_values = self.value.reshape(-1)
            return Diagonal(diag_values)
        elif isinstance(self.value, numbers.Number):
            siz = jnp.prod(shape)
            return Diagonal(self.value * jnp.ones(siz))


class Diff(LinearOperator):
    DEFAULT_ACC = 2

    def __init__(self, dim, axis: GridAxis = None, acc=DEFAULT_ACC):
        """Initializes a Diff instance.

        Parameters
        ----------
        dim: int
            The 0-based index of the axis along which to take the derivative.
        axis: GridAxis
            The grid axis.
        acc: (optional) int
            The accuracy order to use. Must be a positive even number.
        """
        super().__init__(shape=(dim, dim), dtype=_dtype)

        self.set_axis(axis)
        self.dim = dim
        self.acc = acc
        self._order = 1
        self._differentiator = None

    def set_grid(self, grid):
        super().set_grid(grid)
        self.set_axis(self.grid.get_axis(self.dim))

    def set_axis(self, axis: GridAxis):
        self._axis = axis
        self._differentiator = None

    @property
    def axis(self):
        return self._axis

    @property
    def grid(self):
        """Returns the grid used."""
        return getattr(self, "_grid", None)

    @property
    def order(self):
        """Returns the order of the derivative."""
        return self._order

    def __call__(self, f, *args, **kwargs):
        """Applies the differential operator."""

        if "acc" in kwargs:
            # allow to pass down new accuracy deep in expression tree
            new_acc = kwargs["acc"]
            if new_acc != self.acc:
                self._differentiator = None
                self.set_accuracy(new_acc)

        if isinstance(f, LinearOperator):
            f = f(*args, **kwargs)

        return self.differentiator(f)

    @property
    def differentiator(self):
        if self._differentiator is None:
            self._differentiator = build_differentiator(self.order, self.axis, self.acc)
        return self._differentiator

    def set_grid(self, grid):  # noqa: F811
        """Sets the grid for the given differential operator expression.

        Parameters
        ----------
        grid: dict | Grid
            Specifies the grid to use. If a dict is given, an equidistant grid
            is assumed and the dict specifies the spacings along the required axes.
        """
        self._grid = make_grid(grid)
        for child in self.children:
            child.set_grid(self._grid)

    def set_accuracy(self, acc):
        """Sets the requested accuracy for the given differential operator expression.

        Parameters
        ----------
        acc: int
            The accuracy order. Must be a positive, even number.
        """
        self.acc = acc
        for child in self.children:
            child.set_accuracy(acc)

    def matrix(self, shape):
        return self.differentiator.matrix(shape)

    def __pow__(self, power):
        """Returns a Diff instance for a higher order derivative."""
        new_diff = Diff(self.dim, self.axis, acc=self.acc)
        new_diff._order *= power
        return new_diff

    def __mul__(self, other):
        if isinstance(other, Diff) and self.dim == other.dim:
            new_diff = Diff(self.dim, self.axis, acc=self.acc)
            new_diff._order += other.order
            return new_diff
        return super().__mul__(other)
