"""Comprehensive tests for grid functionality."""

import jax.numpy as jnp
import pytest

from fdx.grids import (
    EquidistantAxis,
    Grid,
    GridAxis,
    NonEquidistantAxis,
    make_axis,
    make_grid,
)


class TestGridAxis:
    """Tests for GridAxis base class."""

    def test_grid_axis_creation(self):
        """Test basic GridAxis creation."""
        axis = GridAxis(dim=0)
        assert axis.dim == 0
        assert axis.periodic is False

    def test_grid_axis_periodic(self):
        """Test GridAxis with periodic boundary."""
        axis = GridAxis(dim=1, periodic=True)
        assert axis.dim == 1
        assert axis.periodic is True

    def test_grid_axis_invalid_dimension(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError):
            GridAxis(dim=-1)

        with pytest.raises(ValueError):
            GridAxis(dim=1.5)


class TestEquidistantAxis:
    """Tests for EquidistantAxis."""

    def test_equidistant_axis_creation(self):
        """Test basic EquidistantAxis creation."""
        axis = EquidistantAxis(dim=0, spacing=0.1)
        assert axis.dim == 0
        assert axis.spacing == 0.1
        assert axis.periodic is False

    def test_equidistant_axis_periodic(self):
        """Test EquidistantAxis with periodic boundary."""
        axis = EquidistantAxis(dim=1, spacing=0.5, periodic=True)
        assert axis.periodic is True
        assert axis.spacing == 0.5

    def test_equidistant_axis_invalid_spacing(self):
        """Test that invalid spacing raises error."""
        with pytest.raises(ValueError):
            EquidistantAxis(dim=0, spacing=0)

        with pytest.raises(ValueError):
            EquidistantAxis(dim=0, spacing=-0.1)


class TestNonEquidistantAxis:
    """Tests for NonEquidistantAxis."""

    def test_non_equidistant_axis_creation(self):
        """Test basic NonEquidistantAxis creation."""
        coords = jnp.array([0, 0.1, 0.3, 0.6, 1.0])
        axis = NonEquidistantAxis(dim=0, coords=coords)
        assert axis.dim == 0
        assert jnp.allclose(axis.coords, coords)

    def test_non_equidistant_axis_periodic(self):
        """Test NonEquidistantAxis with periodic boundary."""
        coords = jnp.array([0, 0.2, 0.5, 0.8])
        axis = NonEquidistantAxis(dim=1, coords=coords, periodic=True)
        assert axis.periodic is True


class TestGrid:
    """Tests for Grid class."""

    def test_grid_creation_single_axis(self):
        """Test Grid creation with single axis."""
        axis = EquidistantAxis(dim=0, spacing=0.1)
        grid = Grid(axis)
        assert 0 in grid.axes
        assert grid.get_axis(0) == axis

    def test_grid_creation_multiple_axes(self):
        """Test Grid creation with multiple axes."""
        axis0 = EquidistantAxis(dim=0, spacing=0.1)
        axis1 = EquidistantAxis(dim=1, spacing=0.2)
        grid = Grid(axis0, axis1)

        assert 0 in grid.axes
        assert 1 in grid.axes
        assert grid.get_axis(0) == axis0
        assert grid.get_axis(1) == axis1

    def test_grid_get_nonexistent_axis(self):
        """Test getting non-existent axis returns None."""
        axis = EquidistantAxis(dim=0, spacing=0.1)
        grid = Grid(axis)
        assert grid.get_axis(1) is None


class TestMakeGrid:
    """Tests for make_grid factory function."""

    def test_make_grid_from_grid(self):
        """Test make_grid with Grid instance returns same grid."""
        axis = EquidistantAxis(dim=0, spacing=0.1)
        grid = Grid(axis)
        result = make_grid(grid)
        assert result is grid

    def test_make_grid_from_dict_simple(self):
        """Test make_grid from simple dict."""
        config = {0: 0.1, 1: 0.2}
        grid = make_grid(config)

        axis0 = grid.get_axis(0)
        axis1 = grid.get_axis(1)

        assert isinstance(axis0, EquidistantAxis)
        assert isinstance(axis1, EquidistantAxis)
        assert axis0.spacing == 0.1
        assert axis1.spacing == 0.2

    def test_make_grid_from_dict_with_periodic(self):
        """Test make_grid from dict with periodic specification."""
        config = {0: {"h": 0.1, "periodic": True}, 1: {"h": 0.2, "periodic": False}}
        grid = make_grid(config)

        axis0 = grid.get_axis(0)
        axis1 = grid.get_axis(1)

        assert axis0.periodic is True
        assert axis1.periodic is False

    def test_make_grid_none(self):
        """Test make_grid with None returns None."""
        result = make_grid(None)
        assert result is None

    def test_make_grid_invalid_type(self):
        """Test make_grid with invalid type raises error."""
        with pytest.raises(TypeError):
            make_grid("invalid")


class TestMakeAxis:
    """Tests for make_axis factory function."""

    def test_make_axis_from_axis(self):
        """Test make_axis with GridAxis instance returns same axis."""
        axis = EquidistantAxis(dim=0, spacing=0.1)
        result = make_axis(0, axis)
        assert result is axis

    def test_make_axis_from_number(self):
        """Test make_axis from number creates EquidistantAxis."""
        axis = make_axis(0, 0.1)
        assert isinstance(axis, EquidistantAxis)
        assert axis.dim == 0
        assert axis.spacing == 0.1
        assert axis.periodic is False

    def test_make_axis_from_number_periodic(self):
        """Test make_axis from number with periodic flag."""
        axis = make_axis(1, 0.2, periodic=True)
        assert isinstance(axis, EquidistantAxis)
        assert axis.periodic is True

    def test_make_axis_from_uniform_array(self):
        """Test make_axis from uniform array creates EquidistantAxis."""
        coords = jnp.linspace(0, 1, 11)
        axis = make_axis(0, coords)

        assert isinstance(axis, EquidistantAxis)
        assert jnp.isclose(axis.spacing, 0.1)

    def test_make_axis_from_nonuniform_array(self):
        """Test make_axis from non-uniform array creates NonEquidistantAxis."""
        coords = jnp.array([0, 0.1, 0.3, 0.6, 1.0])
        axis = make_axis(0, coords)

        assert isinstance(axis, NonEquidistantAxis)
        assert jnp.allclose(axis.coords, coords)

    def test_make_axis_from_single_value_array(self):
        """Test make_axis from single-value array."""
        coords = jnp.array([0.5])
        axis = make_axis(0, coords)

        assert isinstance(axis, EquidistantAxis)
        assert axis.spacing == 0.5

    def test_make_axis_invalid_type(self):
        """Test make_axis with invalid type raises error."""
        with pytest.raises(TypeError):
            make_axis(0, "invalid")


class TestGridIntegration:
    """Integration tests for grid functionality."""

    def test_grid_equidistant_2d(self):
        """Test 2D equidistant grid creation."""
        config = {0: 0.1, 1: 0.2}
        grid = make_grid(config)

        assert grid.get_axis(0).spacing == 0.1
        assert grid.get_axis(1).spacing == 0.2

    def test_grid_mixed_periodic(self):
        """Test grid with mixed periodic/non-periodic axes."""
        config = {0: {"h": 0.1, "periodic": True}, 1: {"h": 0.2, "periodic": False}}
        grid = make_grid(config)

        assert grid.get_axis(0).periodic is True
        assert grid.get_axis(1).periodic is False

    def test_grid_3d(self):
        """Test 3D grid creation."""
        axis0 = EquidistantAxis(dim=0, spacing=0.1)
        axis1 = EquidistantAxis(dim=1, spacing=0.2)
        axis2 = EquidistantAxis(dim=2, spacing=0.3)
        grid = Grid(axis0, axis1, axis2)

        assert len(grid.axes) == 3
        assert grid.get_axis(0).spacing == 0.1
        assert grid.get_axis(1).spacing == 0.2
        assert grid.get_axis(2).spacing == 0.3
