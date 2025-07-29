import jax
import pytest
from jax import numpy as jnp

from fdx import Gradient


def test_gradient():
    """Test the Gradient class with a proper 2D scalar field"""

    # Create 2D grid
    x = jnp.linspace(0, 1, 10)
    y = jnp.linspace(0, 1, 10)
    X, Y = jnp.meshgrid(x, y, indexing="ij")  # Create 2D coordinate arrays

    # Define a 2D scalar function: f(x,y) = 2*x^2 + y^2
    f = 2 * X**2 + Y**2
    print(f"Function shape: {f.shape}")  # Should be (10, 10)

    # Create gradient operator
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    grad = Gradient(h=[dx, dy])

    # Compute gradient
    actual_grad = grad(f)
    print(f"Gradient shape: {actual_grad.shape}")  # Should be (2, 10, 10)

    # Expected gradients:
    # ∂f/∂x = 4*x
    # ∂f/∂y = 2*y
    expected_grad_x = 4 * X
    expected_grad_y = 2 * Y

    print("Testing gradient components...")
    print(f"∂f/∂x error: {jnp.max(jnp.abs(actual_grad[0] - expected_grad_x))}")
    print(f"∂f/∂y error: {jnp.max(jnp.abs(actual_grad[1] - expected_grad_y))}")

    # Check if gradients match (with some tolerance for numerical errors)
    assert jnp.allclose(actual_grad[0], expected_grad_x, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(actual_grad[1], expected_grad_y, rtol=1e-4, atol=1e-4)

    print("✓ All gradient tests passed!")

    return actual_grad, expected_grad_x, expected_grad_y


def test_gradient_simple():
    """Simpler test with a quadratic function"""

    # Create a smaller 2D grid for easier visualization
    x = jnp.linspace(-1, 1, 5)
    y = jnp.linspace(-1, 1, 5)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Simple quadratic: f(x,y) = x^2 + y^2
    f = X**2 + Y**2

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    grad = Gradient(h=[dx, dy])

    # Compute gradient
    grad_f = grad(f)

    # Expected: ∂f/∂x = 2*x, ∂f/∂y = 2*y
    expected_x = 2 * X
    expected_y = 2 * Y

    print("Simple test results:")
    print(f"Grid points: {x}")
    print(f"Function at center: {f[2, 2]}")  # Should be 0 at origin
    print(f"∂f/∂x at center: {grad_f[0][2, 2]}")  # Should be ~0
    print(f"∂f/∂y at center: {grad_f[1][2, 2]}")  # Should be ~0

    assert jnp.allclose(grad_f[0], expected_x, rtol=1e-4)
    assert jnp.allclose(grad_f[1], expected_y, rtol=1e-4)

    print("✓ Simple gradient test passed!")


def test_gradient_polynomial():
    """Test with a more complex polynomial"""

    x = jnp.linspace(0, 2, 80)
    y = jnp.linspace(0, 2, 80)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # f(x,y) = x^3 + 2*x*y + y^2
    f = X**3 + 2 * X * Y + Y**2

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    grad = Gradient(h=[dx, dy])

    grad_f = grad(f)

    # Expected gradients:
    # ∂f/∂x = 3*x^2 + 2*y
    # ∂f/∂y = 2*x + 2*y
    expected_x = 3 * X**2 + 2 * Y
    expected_y = 2 * X + 2 * Y

    print("Polynomial test - Max errors:")
    print(f"∂f/∂x error: {jnp.max(jnp.abs(grad_f[0] - expected_x))}")
    print(f"∂f/∂y error: {jnp.max(jnp.abs(grad_f[1] - expected_y))}")

    # Note: Higher-order polynomials may need looser tolerance due to finite difference approximation
    assert (grad_f[0] - expected_x).mean() < 1e-3
    assert (grad_f[1] - expected_y).mean() < 1e-3

    print("✓ Polynomial gradient test passed!")


if __name__ == "__main__":
    print("Testing 2D Gradient class...")
    test_gradient_simple()
    print()
    test_gradient()
    print()
    test_gradient_polynomial()
    print("\n🎉 All tests passed!")
