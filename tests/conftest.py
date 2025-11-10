"""Shared fixtures and configuration for fdx tests."""

import warnings

import jax.numpy as jnp
import pytest

# Filter out Pydantic warnings from third-party dependencies
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
    message=".*UnsupportedFieldAttributeWarning.*",
)
warnings.filterwarnings(
    "ignore",
    message="The 'repr' attribute with value .* was provided to the Field\\(\\) function",
)
warnings.filterwarnings(
    "ignore",
    message="The 'frozen' attribute with value .* was provided to the Field\\(\\) function",
)


@pytest.fixture
def small_grid_1d():
    """Small 1D grid for basic tests."""
    x = jnp.linspace(0, 1, 20)
    dx = x[1] - x[0]
    return x, dx


@pytest.fixture
def medium_grid_1d():
    """Medium 1D grid for accuracy tests."""
    x = jnp.linspace(0, 2 * jnp.pi, 100)
    dx = x[1] - x[0]
    return x, dx


@pytest.fixture
def small_grid_2d():
    """Small 2D grid for basic tests."""
    x = jnp.linspace(0, 1, 20)
    y = jnp.linspace(0, 1, 20)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


@pytest.fixture
def medium_grid_2d():
    """Medium 2D grid for accuracy tests."""
    x = jnp.linspace(0, 2, 40)
    y = jnp.linspace(0, 2, 40)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


@pytest.fixture
def small_grid_3d():
    """Small 3D grid for basic tests."""
    x = jnp.linspace(-1, 1, 20)
    y = jnp.linspace(-1, 1, 20)
    z = jnp.linspace(-1, 1, 20)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z, dx, dy, dz


@pytest.fixture(params=[2, 4, 6, 8])
def accuracy_order(request):
    """Parametrize accuracy orders for tests."""
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def derivative_order(request):
    """Parametrize derivative orders for tests."""
    return request.param


def tolerance_for_accuracy(acc, deriv=1):
    """Calculate tolerance based on accuracy order and derivative order.

    Parameters
    ----------
    acc : int
        Accuracy order
    deriv : int
        Derivative order

    Returns
    -------
    float
        Appropriate tolerance for numerical comparisons
    """
    # Higher derivative orders need more lenient tolerances
    base_tol = 10 ** (-acc / 2)
    return base_tol * (10 ** (deriv - 1))
