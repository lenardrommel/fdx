"""Vector calculus operators built from scalar finite differences."""

from typing import Any, List, Optional, Union

import jax
from jax import numpy as jnp

from .compatible import FinDiff


class VectorOperator:
    """Base class for all vector differential operators.

    Shall not be instantiated directly, but through the child classes.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor for the VectorOperator base class.

        kwargs:
        -------

        h       list with the grid spacings of an N-dimensional uniform grid

        coords  list of 1D arrays with the coordinate values along the N axes.
                This is used for non-uniform grids.

        Either specify "h" or "coords", not both.

        """
        if "acc" in kwargs:
            self.acc = kwargs.pop("acc")
        else:
            self.acc = 2

        if "spac" in kwargs or "h" in kwargs:  # necessary for backward compatibility 0.5.2 => 0.6
            if "spac" in kwargs:
                kw = "spac"
            else:
                kw = "h"
            self.h = kwargs.pop(kw)
            self.ndims = len(self.h)
            self.components = [FinDiff(k, self.h[k], 1) for k in range(self.ndims)]

        if "coords" in kwargs:
            coords = kwargs.pop("coords")
            self.ndims = self.__get_dimension(coords)
            self.components = [FinDiff((k, coords[k], 1), **kwargs) for k in range(self.ndims)]

    def __get_dimension(self, coords: List[jnp.ndarray]) -> int:
        return len(coords)


class Gradient(VectorOperator):
    r"""
    The N-dimensional gradient.

    .. math::
        \nabla = \left(\frac{\partial}{\partial x_0}, \frac{\partial}{\partial x_1},
        ... , \frac{\partial}{\partial x_{N-1}}\right)

    :param kwargs:  exactly one of *h* and *coords* must be specified

             *h*
                     list with the grid spacings of an N-dimensional uniform grid
             *coords*
                     list of 1D arrays with the coordinate values along the N axes.
                     This is used for non-uniform grids.

             *acc*
                     accuracy order, must be positive integer, default is 2
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(self, f: jnp.ndarray, axis: Optional[int] = None, has_batch: bool = False) -> jnp.ndarray:
        """
        Applies the N-dimensional gradient to the array f.

        :param f:  ``jax.Array`` Array to apply the gradient to. It represents a scalar function,
                so it must have N axes for the N independent variables.
        :param axis: Optional[int]
                If None (default), compute the full gradient (all partial derivatives).
                If an integer, compute the derivative along the specified axis only.

        :param has_batch: bool
                If True, the first axis of f is considered a batch dimension.


        :returns: ``jax.Array``

                The gradient of f, which has N+1 axes, i.e. it is
                an array of N arrays of N axes each.

        """
        if not isinstance(f, jnp.ndarray):
            raise TypeError("Function to differentiate must be jnp.ndarray")

        if has_batch:
            # scalar field per batch item must have exactly ndims axes
            if axis is None:
                if f.ndim != self.ndims + 1:
                    raise ValueError("With has_batch=True and axis=None, expected shape (batch, *spatial)")

                # vmap over batch, compute full gradient on each sample
                def grad_one(sample):
                    parts = [comp(sample, acc=self.acc) for comp in self.components]  # each: (*spatial)
                    return jnp.stack(parts, axis=0)  # (ndims, *spatial)

                return jax.vmap(grad_one)(f)  # (batch, ndims, *spatial)

            # axis derivative with batch: axis refers to spatial axis (0..ndims-1)
            s_axis = int(axis) % self.ndims

            def deriv_one(sample):
                return self.components[s_axis](sample, acc=self.acc)  # (*spatial)

            return jax.vmap(deriv_one)(f)  # (batch, *spatial)

        # no batch
        if axis is None:
            if f.ndim != self.ndims:
                raise ValueError("Gradients can only be applied to scalar functions")
            parts = [comp(f, acc=self.acc) for comp in self.components]
            return jnp.stack(parts, axis=0)  # (ndims, *spatial)

        s_axis = int(axis) % self.ndims
        return self.components[s_axis](f, acc=self.acc)  # (*spatial)


class Divergence(VectorOperator):
    r"""
    The N-dimensional divergence.

    .. math::

       {\rm \bf div} = \nabla \cdot

    :param kwargs:  exactly one of *h* and *coords* must be specified

         *h*
                 list with the grid spacings of an N-dimensional uniform grid
         *coords*
                 list of 1D arrays with the coordinate values along the N axes.
                 This is used for non-uniform grids.

         *acc*
                 accuracy order, must be positive integer, default is 2

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the divergence to the array f.

        :param f: ``numpy.ndarray``

               a vector function of N variables, so its array has N+1 axes.

        :returns: ``numpy.ndarray``

               the divergence, which is a scalar function of N variables, so it's array dimension has N axes

        """
        if not isinstance(f, jnp.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be jnp.ndarray or list of jnp.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Divergence can only be applied to vector functions of the same dimension")

        result = jnp.zeros(f.shape[1:])

        result = jnp.sum(
            jnp.stack([self.components[k](f[k], acc=self.acc) for k in range(self.ndims)]),
            axis=0,
        )

        return result


class Curl(VectorOperator):
    r"""
    The curl operator.

    .. math::

        {\rm \bf rot} = \nabla \times

    Is only defined for 3D.

    :param kwargs:  exactly one of *h* and *coords* must be specified

     *h*
             list with the grid spacings of a 3-dimensional uniform grid
     *coords*
             list of 1D arrays with the coordinate values along the 3 axes.
             This is used for non-uniform grids.

     *acc*
             accuracy order, must be positive integer, default is 2


    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if self.ndims != 3:
            raise ValueError(f"Curl operation is only defined in 3 dimensions. {self.ndims} were given.")

    def __call__(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the curl to the array f.

        :param f: ``numpy.ndarray``

               a vector function of N variables, so its array has N+1 axes.

        :returns: ``numpy.ndarray``

               the curl, which is a vector function of N variables, so it's array dimension has N+1 axes

        """
        if not isinstance(f, jnp.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be jnp.ndarray or list of jnp.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Curl can only be applied to vector functions of the three dimensions")

        result = jnp.zeros(f.shape)

        result = result.at[0].add(self.components[1](f[2], acc=self.acc) - self.components[2](f[1], acc=self.acc))
        result = result.at[1].add(self.components[2](f[0], acc=self.acc) - self.components[0](f[2], acc=self.acc))
        result = result.at[2].add(self.components[0](f[1], acc=self.acc) - self.components[1](f[0], acc=self.acc))

        return result


class Laplacian(VectorOperator):
    """N-dimensional Laplacian operator for scalar fields."""

    def __init__(self, h: Optional[List[float]] = None, acc: int = 2) -> None:
        """Create a Laplacian operator.

        Parameters
        ----------
        h
            Grid spacings for a uniform grid. If not provided, defaults to `[1.0]`.
        acc
            Accuracy order (positive integer).
        """
        h_list = h or [1.0]
        h_arr = wrap_in_ndarray(h_list)

        self._parts = [FinDiff((k, h_arr[k], 2), acc=acc) for k in range(len(h_arr))]
        super().__init__(h=h_arr, acc=acc)

    def __call__(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the Laplacian to the array f.

        :param f: ``numpy.ndarray``

               a scalar function of N variables, so its array has N axes.

        :returns: ``numpy.ndarray``

               the Laplacian of f, which is a scalar function of N variables, so it's array dimension has N axes

        """
        laplace_f = jnp.zeros_like(f)

        for part in self._parts:
            laplace_f = laplace_f + part(f)

        return laplace_f


class Jacobian(VectorOperator):
    """
    Jacobian of a vector-valued field with respect to spatial variables.

    No batch:  u.shape = (*spatial, *components)  ->  J.shape = (ndims, *spatial, *components)
    Batch:     u.shape = (batch, *spatial, *components) -> J.shape = (batch, ndims, *spatial, *components)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(self, u: jnp.ndarray, has_batch: bool = False) -> jnp.ndarray:
        """Compute the Jacobian of a vector-valued field.

        Parameters
        ----------
        u
            Field with shape `(*spatial, *components)` or `(batch, *spatial, *components)`
            when `has_batch=True`.
        has_batch
            Whether the first axis is a batch axis.

        Returns
        -------
        jax.Array
            Jacobian array with shape `(ndims, *spatial, *components)` or
            `(batch, ndims, *spatial, *components)` when `has_batch=True`.
        """
        if not isinstance(u, jnp.ndarray):
            raise TypeError("Function to differentiate must be jnp.ndarray")

        if has_batch:
            if u.ndim < self.ndims + 1:
                raise ValueError("Expected u.shape = (batch, *spatial, *components)")
            batch = u.shape[0]
            spatial_shape = u.shape[1 : 1 + self.ndims]
            comp_shape = u.shape[1 + self.ndims :]
            if len(spatial_shape) != self.ndims:
                raise ValueError("Spatial dims do not match operator ndims")

            # flatten components: (B, *S, *C) -> (B, *S, Cflat)
            u_flat = u.reshape(batch, *spatial_shape, -1)
            # move component axis in front for vmap: (B, *S, C) -> (B, C, *S)
            u_flat = jnp.moveaxis(u_flat, -1, 1)

            def jac_one_component(field_1comp):
                # field_1comp: (*S)
                parts = [self.components[ax](field_1comp, acc=self.acc) for ax in range(self.ndims)]
                return jnp.stack(parts, axis=0)  # (ndims, *S)

            # vmap over components, then over batch
            J = jax.vmap(lambda u_c: jax.vmap(jac_one_component)(u_c))(u_flat)
            # J: (B, C, ndims, *S)
            J = jnp.moveaxis(J, 1, -1)  # (B, ndims, *S, C)
            return J.reshape(batch, self.ndims, *spatial_shape, *comp_shape)

        else:
            if u.ndim < self.ndims:
                raise ValueError("Expected u.shape = (*spatial, *components)")
            spatial_shape = u.shape[: self.ndims]
            comp_shape = u.shape[self.ndims :]
            if len(spatial_shape) != self.ndims:
                raise ValueError("Spatial dims do not match operator ndims")

            u_flat = u.reshape(*spatial_shape, -1)  # (*S, Cflat)
            u_flat = jnp.moveaxis(u_flat, -1, 0)  # (Cflat, *S)

            def jac_one_component(field_1comp):
                parts = [self.components[ax](field_1comp, acc=self.acc) for ax in range(self.ndims)]
                return jnp.stack(parts, axis=0)  # (ndims, *S)

            Jc = jax.vmap(jac_one_component)(u_flat)  # (Cflat, ndims, *S)
            J = jnp.moveaxis(Jc, 0, -1)  # (ndims, *S, Cflat)
            return J.reshape(self.ndims, *spatial_shape, *comp_shape)


def wrap_in_ndarray(value: Union[jnp.ndarray, List[float]]) -> jnp.ndarray:
    """Wraps the argument in a numpy.ndarray.

    If value is a scalar, it is converted in a list first.
    If value is array-like, the shape is conserved.

    """
    if hasattr(value, "__len__"):
        return jnp.array(value)
    else:
        return jnp.array([value])
