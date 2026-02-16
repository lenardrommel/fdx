"""fdx: finite-difference operators for JAX arrays."""

from ._version import __version__

from fdx.coefs import coefficients
from fdx.compatible import Coef, Coefficient, FinDiff, Id
from fdx.config import set_dtype
from fdx.interface import Diff
from fdx.vector import Curl, Divergence, Gradient, Jacobian, Laplacian

__all__ = [
    "set_dtype",
    "coefficients",
    "Coefficient",
    "Coef",
    "FinDiff",
    "Id",
    "Diff",
    "Curl",
    "Divergence",
    "Gradient",
    "Laplacian",
    "Jacobian",
]
