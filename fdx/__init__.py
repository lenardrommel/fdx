__version__ = "0.1.0"

from fdx.coefs import coefficients
from fdx.compatible import FinDiff
from fdx.config import set_dtype
from fdx.interface import Diff
from fdx.vector import Curl, Divergence, Gradient, Laplacian

__all__ = [
    "set_dtype",
    "coefficients",
    "FinDiff",
    "Diff",
    "Curl",
    "Divergence",
    "Gradient",
    "Laplacian",
]
