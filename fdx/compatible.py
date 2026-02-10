"""Compatibility layer for the `findiff`-style public API."""

from fdx.interface import Diff
from fdx.operators import FieldOperator, Identity


def FinDiff(*args, **kwargs):
    """Construct a finite-difference expression compatible with `findiff.FinDiff`.

    Parameters
    ----------
    *args
        Either a single tuple `(axis, spacing_or_coords, order)`/`(axis, spacing_or_coords)`,
        or multiple such tuples to create mixed derivatives via multiplication.
    **kwargs
        Passed through to `fdx.interface.Diff`.

    Returns
    -------
    Expression
        An operator expression that can be applied to arrays.
    """
    if len(args) > 3:
        raise ValueError("FinDiff accepts not more than 3 positional arguments.")

    def diff_from_tuple(tpl):
        if len(tpl) == 3:
            axis, h, order = tpl
            return Diff(axis, h, **kwargs) ** order
        elif len(tpl) == 2:
            axis, h = tpl
            return Diff(axis, h, **kwargs)

    if isinstance(args[0], (list, tuple)):
        diffs = []
        for tpl in args:
            diffs.append(diff_from_tuple(tpl))
        fd = diffs[0]
        for diff in diffs[1:]:
            fd = fd * diff
        return fd

    return diff_from_tuple(args)


# Aliases for compatibility with findiff API
class Coefficient(FieldOperator):
    """Variable coefficient operator (pointwise multiplication).

    Compatible alias with findiff's Coefficient; wraps FieldOperator.
    """

    pass


Coef = Coefficient
Id = Identity
