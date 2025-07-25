from jax import numpy as jnp

_dtype = jnp.float32

def set_dtype(dtype):
    """Set the default dtype for finite difference calculations."""
    global _dtype
    _dtype = dtype