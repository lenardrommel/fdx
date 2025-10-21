import jax
from jax import numpy as jnp
from linox._matrix import Identity

from .vector import Gradient, Laplacian


class Norm:
    """Base class for norm operators."""

    def __init__(self, op, **kwargs):
        """Contructor for the Norm class.

        Parameters
        ----------
        op : Operator for the Sobolev space norm. For example, a Gradient or Laplacian operator.
        kwargs : dict

        """
        self.op = op
        self.kwargs = kwargs

    def __call__(self, x, y, *args, **kwargs):
        """Compute the norm of the field `f`."""
        raise NotImplementedError("Norm must implement __call__ method.")


class L2Norm(Norm):
    def __init__(self, **kwargs):
        op = Identity
        super().__init__(op, **kwargs)

    def __call__(self, x, y, *args, **kwargs):
        return jnp.sqrt(jnp.sum(jnp.square(y - x), axis=-1))


class H1Norm(Norm):
    """Implements the Sobolev H1 norm."""

    def __init__(self, op, **kwargs):
        op = Gradient(op, **kwargs)
        super().__init__(op, **kwargs)

    def __call__(self, x, y, *args, **kwargs):
        return jnp.sqrt(
            jnp.sum(jnp.square(y - x), axis=-1)
            + jnp.sum(
                jnp.square(self.op(x, *args, **kwargs) - self.op(y, *args, **kwargs)),
                axis=-1,
            )
        )
