# __init__.py

"""fdx: finite-difference operators for JAX arrays."""

from importlib.metadata import PackageNotFoundError, version

try:
    from fdx._version import __version__
except ImportError:
    try:
        __version__ = version("fdx")
    except PackageNotFoundError:
        __version__ = "0.0.0"

from fdx.coefs import coefficients
from fdx.compatible import Coef, Coefficient, FinDiff, Id
from fdx.config import set_dtype, set_x64
from fdx.interface import Diff
from fdx.vector import Curl, Divergence, Gradient, Jacobian, Laplacian

__all__ = [
    "Coef",
    "Coefficient",
    "Curl",
    "Diff",
    "Divergence",
    "FinDiff",
    "Gradient",
    "Id",
    "Jacobian",
    "Laplacian",
    "coefficients",
    "set_dtype",
    "set_x64",
]
