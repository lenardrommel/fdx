from jax import numpy as jnp

_dtype = jnp.float32


def set_dtype(dtype):
    """Set the default dtype for finite difference calculations."""
    global _dtype
    _dtype = dtype


def get_precision():
    """Get the current default dtype for finite difference calculations."""
    if _dtype == jnp.float32:
        return 1.0e-14
    elif _dtype == jnp.float64:
        return 1.0e-28
