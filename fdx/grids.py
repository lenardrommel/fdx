# grids.py

from typing import Dict, Optional, Union

import jax
from jax import numpy as jnp


class GridAxis:
    def __init__(self, dim: int, periodic: bool = False) -> None:
        if not isinstance(dim, int):
            raise ValueError("Dimension must be an integer")
        if dim < 0:
            raise ValueError("Dimension must be >= 0.")
        self.dim = dim
        self.periodic = periodic


class EquidistantAxis(GridAxis):
    def __init__(self, dim: int, spacing: float, periodic: bool = False) -> None:
        super().__init__(dim, periodic)
        if spacing <= 0:
            raise ValueError("Spacing must be > 0.")
        self.spacing = spacing


class NonEquidistantAxis(GridAxis):
    def __init__(self, dim: int, coords: jnp.ndarray, periodic: bool = False) -> None:
        super().__init__(dim, periodic)
        self.coords = coords


class Grid:
    def __init__(self, *axes: GridAxis) -> None:
        self.axes: Dict[int, GridAxis] = {ax.dim: ax for ax in axes}

    def get_axis(self, dim: int) -> Optional[GridAxis]:
        return self.axes.get(int(dim))


def make_grid(config_or_grid: Union[Grid, dict, None]) -> Optional[Grid]:
    """Makes or returns a grid based on configuration or an actual Grid instance.

    Historically, the API allowed to specify grid using a variety of
    shortcuts using a single number, dicts or full Grid instances.

    The purpose of this function is to keep other modules closed with
    respect of addition an modification of GridAxis and Grid types.
    """
    if isinstance(config_or_grid, Grid):
        return config_or_grid
    elif isinstance(config_or_grid, dict):
        config = config_or_grid
        axes = []
        for dim, ax_config in config.items():
            if isinstance(ax_config, dict):
                ax = EquidistantAxis(
                    dim, ax_config["h"], periodic=ax_config.get("periodic", False)
                )
            else:
                ax = EquidistantAxis(dim, ax_config)
            axes.append(ax)
        return Grid(*axes)
    elif config_or_grid is None:
        return None
    else:
        raise TypeError(f"Unsupported grid type: {type(config_or_grid)}")


def make_axis(
    dim: int,
    config_or_axis: Union[GridAxis, float, int, jax.Array],
    periodic: bool = False,
) -> GridAxis:
    """Makes or returns a grid axis based on configuration or an actual GridAxis instance.

    Historically, the API allowed to specify axes using a variety of
    shortcuts using a single number, dicts or full GridAxis instances.

    The purpose of this function is to keep other modules closed with
    respect of addition an modification of GridAxis and Grid types.
    """

    if isinstance(config_or_axis, GridAxis):
        return config_or_axis
    if isinstance(config_or_axis, (int, float)):
        return EquidistantAxis(dim, spacing=float(config_or_axis), periodic=periodic)
    if isinstance(config_or_axis, jax.Array):
        if config_or_axis.size > 1:
            spacing = jnp.diff(config_or_axis)
            if jnp.allclose(spacing, spacing[0]):
                return EquidistantAxis(dim, spacing=spacing[0], periodic=periodic)
            return NonEquidistantAxis(dim, coords=config_or_axis, periodic=periodic)
        else:
            return EquidistantAxis(
                dim, spacing=config_or_axis.item(), periodic=periodic
            )
    else:
        raise TypeError(
            f"Unsupported axis type: {type(config_or_axis)}. "
            "Expected GridAxis, number, or jax.Array."
        )
    # elif isinstance(config_or_axis, jnp.ndarray):
    #     return NonEquidistantAxis(dim, coords=config_or_axis, periodic=periodic)


def set_accuracy(accuracy: float) -> None:
    """
    Placeholder for setting accuracy. Not yet implemented.
    """
    raise NotImplementedError("set_accuracy is not implemented yet.")
