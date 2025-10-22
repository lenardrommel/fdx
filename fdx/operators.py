"""operators.py: linox-free differential operator expressions.

This module provides a small expression system compatible with JAX that
implements differential operators (Diff) and simple algebra (Add/Mul),
without depending on external linear-operator packages.
"""

import numbers
from abc import ABC, abstractmethod
from typing import List, Optional

from jax import numpy as jnp

from fdx.findiff import build_differentiator
from fdx.grids import GridAxis, make_grid
from fdx.stencils import StencilSet


class Expression(ABC):
    """Base class for differential operator expressions."""

    __array_priority__ = 100

    def __init__(self, *args, **kwargs) -> None:
        self.children: List[Expression] = []  # type: ignore[name-defined]

    @abstractmethod
    def __call__(self, f, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def matrix(self, shape):
        raise NotImplementedError

    def stencil(self, shape):
        return StencilSet(self, shape)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Add(ScalarOperator(-1) * other, self)

    def __rsub__(self, other):
        return Add(ScalarOperator(-1) * other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    @property
    def grid(self):
        return getattr(self, "_grid", None)

    def set_grid(self, grid):
        self._grid = make_grid(grid)
        for child in self.children:
            child.set_grid(self._grid)

    def set_accuracy(self, acc: int):
        self.acc = acc  # type: ignore[attr-defined]
        for child in self.children:
            child.set_accuracy(acc)


class FieldOperator(Expression):
    """Pointwise multiplication operator."""

    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def __call__(self, f, *args, **kwargs):
        if isinstance(f, (numbers.Number, jnp.ndarray)):
            return self.value * f
        if isinstance(f, Expression):
            return self.value * f(*args, **kwargs)
        return self.value * f

    def matrix(self, shape):
        siz = int(jnp.prod(jnp.array(shape)))
        if isinstance(self.value, jnp.ndarray):
            diag_values = self.value.reshape(-1)
        elif isinstance(self.value, numbers.Number):
            diag_values = jnp.full((siz,), float(self.value))
        else:
            raise TypeError("Unsupported field value type for matrix()")
        return jnp.diag(diag_values)


class ScalarOperator(FieldOperator):
    def __init__(self, value):
        if not isinstance(value, numbers.Number):
            raise ValueError(f"Expected number, got {type(value)}")
        super().__init__(value)

    def matrix(self, shape):
        siz = int(jnp.prod(jnp.array(shape)))
        return jnp.eye(siz) * float(self.value)


class Identity(ScalarOperator):
    def __init__(self):
        super().__init__(1.0)


class BinaryOperation(Expression):
    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]


class Add(BinaryOperation):
    def __init__(self, left, right):
        if isinstance(left, (numbers.Number, jnp.ndarray)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, jnp.ndarray)):
            right = FieldOperator(right)
        super().__init__()
        self.children = [left, right]

    def __call__(self, f, *args, **kwargs):
        return self.left(f, *args, **kwargs) + self.right(f, *args, **kwargs)

    def matrix(self, shape):
        return self.left.matrix(shape) + self.right.matrix(shape)


class Mul(BinaryOperation):
    def __init__(self, left, right):
        if isinstance(left, (numbers.Number, jnp.ndarray)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, jnp.ndarray)):
            right = FieldOperator(right)
        super().__init__()
        self.children = [left, right]

    def __call__(self, f, *args, **kwargs):
        return self.left(self.right(f, *args, **kwargs), *args, **kwargs)

    def matrix(self, shape):
        return self.left.matrix(shape) @ self.right.matrix(shape)


class Diff(Expression):
    DEFAULT_ACC = 2

    def __init__(self, dim, axis: Optional[GridAxis] = None, acc=DEFAULT_ACC):
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
        super().__init__()
        self.dim = dim
        self.acc = acc
        self._order = 1
        self._axis: Optional[GridAxis] = None
        self._differentiator = None
        self.set_axis(axis)

    def set_grid(self, grid):
        super().set_grid(grid)
        if self.grid is not None:
            self.set_axis(self.grid.get_axis(self.dim))

    def set_axis(self, axis: Optional[GridAxis]):
        self._axis = axis
        self._differentiator = None

    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    def __call__(self, f, *args, **kwargs):
        if "acc" in kwargs:
            new_acc = kwargs["acc"]
            if new_acc != self.acc:
                self._differentiator = None
                self.set_accuracy(new_acc)

        if isinstance(f, Expression):
            f = f(*args, **kwargs)

        return self.differentiator(f)

    @property
    def differentiator(self):
        if self._differentiator is None:
            if self._axis is None:
                raise ValueError("Axis is not set for Diff operator.")
            self._differentiator = build_differentiator(self.order, self.axis, self.acc)
        return self._differentiator

    def matrix(self, shape):
        return self.differentiator.matrix(shape)

    def __pow__(self, power):
        new_diff = Diff(self.dim, self.axis, acc=self.acc)
        new_diff._order *= power
        return new_diff

    def __mul__(self, other):
        if isinstance(other, Diff) and self.dim == other.dim:
            new_diff = Diff(self.dim, self.axis, acc=self.acc)
            new_diff._order += other.order
            return new_diff
        return super().__mul__(other)
