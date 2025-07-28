__version__ = "0.1.0"

from fdx.coefs import coefficients
from fdx.compatible import FinDiff
from fdx.config import set_dtype
from fdx.interface import Diff

__all__ = ["set_dtype", "coefficients", "FinDiff", "Diff"]
