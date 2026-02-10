"""Global configuration for finite-difference coefficient computations."""

from jax import numpy as jnp

_dtype = jnp.float64


def set_dtype(dtype: jnp.dtype) -> None:
    """Set the default dtype for coefficient computations.

    Parameters
    ----------
    dtype
        Default dtype used in coefficient computations.
    """
    global _dtype
    _dtype = dtype


def get_precision() -> float:
    """Return the numerical precision used for coefficient computations.

    Returns
    -------
    float
        A conservative tolerance associated with the current default dtype.
    """
    if _dtype == jnp.float32:
        return 1.0e-14
    elif _dtype == jnp.float64:
        return 1.0e-28
    else:
        raise ValueError("Unknown dtype.")
