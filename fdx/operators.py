"""operators.py: linox-free differential operator expressions.

This module provides a small expression system compatible with JAX that
implements differential operators (Diff) and simple algebra (Add/Mul),
without depending on external linear-operator packages.
"""

import numbers
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import jax
from jax import numpy as jnp

from fdx.findiff import build_differentiator
from fdx.grids import GridAxis, make_grid
from fdx.stencils import StencilSet
from fdx.types import Array


class Expression(ABC):
    """Base class for differential operator expressions."""

    __array_priority__ = 100

    def __init__(self, *args, **kwargs) -> None:
        self.children: List[Expression] = []  # type: ignore[name-defined]

    @abstractmethod
    def __call__(self, f, *args, **kwargs):
        """Evaluate the expression on an input field."""
        raise NotImplementedError

    @abstractmethod
    def matrix(self, shape):
        """Return a dense matrix representation for a given field shape."""
        raise NotImplementedError

    def stencil(self, shape):
        """Return a `StencilSet` representation for a given field shape.

        Parameters
        ----------
        shape
            Shape of the discretized field.

        Returns
        -------
        StencilSet
            A set of stencils compatible with this expression on `shape`.
        """
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
        """Optional `Grid` attached to this expression tree."""
        return getattr(self, "_grid", None)

    def set_grid(self, grid):
        """Attach a grid to this expression and all children.

        Parameters
        ----------
        grid
            A grid specification accepted by `fdx.grids.make_grid`.
        """
        self._grid = make_grid(grid)
        for child in self.children:
            child.set_grid(self._grid)

    def set_accuracy(self, acc: int):
        """Set the finite-difference accuracy order for this expression.

        Parameters
        ----------
        acc
            Positive even accuracy order.
        """
        self.acc = acc  # type: ignore[attr-defined]
        for child in self.children:
            child.set_accuracy(acc)


@jax.tree_util.register_pytree_node_class
class FieldOperator(Expression):
    """Pointwise multiplication operator."""

    def __init__(self, value: Union[float, Array]) -> None:
        super().__init__()
        self.value = value

    def __call__(self, f, *args, **kwargs):
        """Apply pointwise multiplication to an array or expression."""
        if isinstance(f, (numbers.Number, Array)):
            return self.value * f
        if isinstance(f, Expression):
            return self.value * f(*args, **kwargs)
        return self.value * f

    def matrix(self, shape):
        """Return a dense diagonal matrix for the pointwise multiplier."""
        siz = int(jnp.prod(jnp.array(shape)))
        if isinstance(self.value, Array):
            diag_values = self.value.reshape(-1)
        elif isinstance(self.value, (int, float)):
            diag_values = jnp.full((siz,), float(self.value))
        else:
            raise TypeError("Unsupported field value type for matrix()")
        return jnp.diag(diag_values)

    def tree_flatten(self):
        """Flatten into (children, aux_data) for JAX pytree."""
        if isinstance(self.value, Array):
            return (self.value,), None
        return (), (self.value,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from (children, aux_data)."""
        if children:
            return cls(children[0])
        return cls(aux_data[0])


@jax.tree_util.register_pytree_node_class
class ScalarOperator(FieldOperator):
    """Scalar multiplication operator."""

    def __init__(self, value: float):
        """Create a scalar multiplication operator.

        Parameters
        ----------
        value
            Scalar multiplier.
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Expected number, got {type(value)}")
        super().__init__(float(value))

    def matrix(self, shape):
        """Return a dense scaled identity matrix."""
        siz = int(jnp.prod(jnp.array(shape)))
        return jnp.eye(siz) * float(self.value)

    def tree_flatten(self):
        """Flatten into (children, aux_data) for JAX pytree."""
        return (), (self.value,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from (children, aux_data)."""
        return cls(aux_data[0])


@jax.tree_util.register_pytree_node_class
class Identity(ScalarOperator):
    """Identity operator."""

    def __init__(self):
        """Create an identity operator."""
        super().__init__(1.0)

    def tree_flatten(self):
        """Flatten into (children, aux_data) for JAX pytree."""
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from (children, aux_data)."""
        return cls()


class BinaryOperation(Expression):
    """Base class for binary expression nodes."""

    @property
    def left(self):
        """Left child expression."""
        return self.children[0]

    @property
    def right(self):
        """Right child expression."""
        return self.children[1]


@jax.tree_util.register_pytree_node_class
class Add(BinaryOperation):
    """Sum of two expressions."""

    def __init__(self, left, right):
        """Create an addition node."""
        if isinstance(left, (numbers.Number, Array)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, Array)):
            right = FieldOperator(right)
        super().__init__()
        self.children = [left, right]

    def __call__(self, f, *args, **kwargs):
        """Evaluate the sum on an input field."""
        return self.left(f, *args, **kwargs) + self.right(f, *args, **kwargs)

    def matrix(self, shape):
        """Return the dense matrix representation of the sum."""
        return self.left.matrix(shape) + self.right.matrix(shape)

    def tree_flatten(self):
        """Flatten into (children, aux_data) for JAX pytree."""
        return (self.left, self.right), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from (children, aux_data)."""
        return cls(children[0], children[1])


@jax.tree_util.register_pytree_node_class
class Mul(BinaryOperation):
    """Composition of two expressions."""

    def __init__(self, left, right):
        """Create a composition node."""
        if isinstance(left, (numbers.Number, Array)):
            left = FieldOperator(left)
        if isinstance(right, (numbers.Number, Array)):
            right = FieldOperator(right)
        super().__init__()
        self.children = [left, right]

    def __call__(self, f, *args, **kwargs):
        """Evaluate the composition on an input field."""
        return self.left(self.right(f, *args, **kwargs), *args, **kwargs)

    def matrix(self, shape):
        """Return the dense matrix representation of the composition."""
        return self.left.matrix(shape) @ self.right.matrix(shape)

    def tree_flatten(self):
        """Flatten into (children, aux_data) for JAX pytree."""
        return (self.left, self.right), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from (children, aux_data)."""
        return cls(children[0], children[1])


@jax.tree_util.register_pytree_node_class
class Diff(Expression):
    """Partial derivative operator along a single grid axis."""

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
        """Attach a grid and update the axis derived from `dim`."""
        super().set_grid(grid)
        if self.grid is not None:
            self.set_axis(self.grid.get_axis(self.dim))

    def set_axis(self, axis: Optional[GridAxis]):
        """Set the underlying `GridAxis` and reset cached differentiator."""
        self._axis = axis
        self._differentiator = None

    @property
    def axis(self):
        """Axis metadata used to build the underlying differentiator."""
        return self._axis

    @property
    def order(self):
        """Derivative order."""
        return self._order

    def __call__(self, f, *args, **kwargs):
        """Apply the derivative to an array or expression."""
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
        """Lazily constructed finite-difference differentiator."""
        if self._differentiator is None:
            if self._axis is None:
                raise ValueError("Axis is not set for Diff operator.")
            self._differentiator = build_differentiator(self.order, self.axis, self.acc)
        return self._differentiator

    def matrix(self, shape):
        """Return the dense matrix representation of the derivative."""
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

    def tree_flatten(self):
        """Flatten into (children, aux_data) for JAX pytree."""
        # The differentiator may contain JAX arrays; include it as a child if it exists
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
