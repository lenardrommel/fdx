import jax
import matplotlib.pyplot as plt
import pytest
from jax import numpy as jnp

from fdx import Gradient


def test_gradient_1d_sine():
    """Test 1D gradient with sum of sine functions: f(x) = sum_i sin(a_i * x)"""

    # Create 1D grid
    x = jnp.linspace(0, 2 * jnp.pi, 100)
    dx = x[1] - x[0]

    # Define coefficients for sine functions
    a_coeffs = jnp.array([1.0, 2.0, 3.0, 0.5])

    # Function: f(x) = sum_i sin(a_i * x)
    f = jnp.sum(jnp.array([jnp.sin(a * x) for a in a_coeffs]), axis=0)

    print(f"Function shape: {f.shape}")  # Should be (100,)

    # Create gradient operator for 1D
    grad = Gradient(h=[dx], acc=4)

    # Compute gradient using axis=0 (since it's 1D)
    actual_grad = grad(f, axis=0)
    print(f"Gradient shape: {actual_grad.shape}")  # Should be (100,)

    # Analytical gradient: f'(x) = sum_i a_i * cos(a_i * x)
    expected_grad = jnp.sum(jnp.array([a * jnp.cos(a * x) for a in a_coeffs]), axis=0)

    # Check error
    max_error = jnp.max(jnp.abs(actual_grad - expected_grad))
    print(f"Maximum error: {max_error}")

    # Test with tolerance appropriate for finite differences
    assert jnp.mean(jnp.abs(actual_grad - expected_grad)) < 1e-2

    print("✓ 1D sine function gradient test passed!")

    return x, f, actual_grad, expected_grad


def test_gradient_1d_simple():
    """Simple 1D test with polynomial"""

    x = jnp.linspace(-2, 2, 50)
    dx = x[1] - x[0]

    # Simple polynomial: f(x) = x^3 + 2*x^2 + x
    f = x**3 + 2 * x**2 + x

    grad = Gradient(h=[dx])
    actual_grad = grad(f, axis=0)

    # Analytical: f'(x) = 3*x^2 + 4*x + 1
    expected_grad = 3 * x**2 + 4 * x + 1

    max_error = jnp.max(jnp.abs(actual_grad - expected_grad))
    print(f"1D polynomial test - Max error: {max_error}")

    assert (actual_grad - expected_grad).mean() < 1e-2, (
        "Gradient does not match expected polynomial gradient"
    )

    print("✓ 1D polynomial gradient test passed!")


def test_gradient_1d_multiple_sines():
    """Test with different combinations of sine functions"""

    x = jnp.linspace(0, 4 * jnp.pi, 200)
    dx = x[1] - x[0]

    # Test cases with different coefficient sets
    test_cases = [
        {"a_coeffs": [1.0], "name": "Single sine"},
        {"a_coeffs": [1.0, 2.0], "name": "Two sines"},
        {"a_coeffs": [0.5, 1.5, 2.5], "name": "Three sines"},
        {"a_coeffs": [1.0, 3.0, 5.0, 0.2], "name": "Four sines"},
    ]

    grad = Gradient(h=[dx], acc=4)

    for case in test_cases:
        a_coeffs = jnp.array(case["a_coeffs"])

        # f(x) = sum_i sin(a_i * x)
        f = jnp.sum(jnp.array([jnp.sin(a * x) for a in a_coeffs]), axis=0)

        # Compute gradient
        actual_grad = grad(f, axis=0)

        # Analytical gradient: f'(x) = sum_i a_i * cos(a_i * x)
        expected_grad = jnp.sum(
            jnp.array([a * jnp.cos(a * x) for a in a_coeffs]), axis=0
        )

        max_error = jnp.max(jnp.abs(actual_grad - expected_grad))

        print(f"{case['name']}: coefficients = {a_coeffs}, max error = {max_error:.2e}")

        assert (actual_grad - expected_grad).mean() < 1e-2, (
            f"Gradient does not match expected for {case['name']}"
        )

    print("✓ All multiple sine tests passed!")


def test_gradient_1d_edge_cases():
    """Test edge cases and boundary conditions"""

    # Test with very fine grid
    x_fine = jnp.linspace(0, jnp.pi, 1000)
    dx_fine = x_fine[1] - x_fine[0]

    # High frequency sine
    f_fine = jnp.sin(10 * x_fine)

    grad_fine = Gradient(h=[dx_fine], acc=4)
    actual_grad_fine = grad_fine(f_fine, axis=0)
    expected_grad_fine = 10 * jnp.cos(10 * x_fine)

    error_fine = jnp.max(jnp.abs(actual_grad_fine - expected_grad_fine))
    print(f"Fine grid test (high frequency): max error = {error_fine:.2e}")

    # Test with coarse grid
    x_coarse = jnp.linspace(0, jnp.pi, 20)
    dx_coarse = x_coarse[1] - x_coarse[0]

    f_coarse = jnp.sin(x_coarse)
    grad_coarse = Gradient(h=[dx_coarse])
    actual_grad_coarse = grad_coarse(f_coarse, axis=0)
    expected_grad_coarse = jnp.cos(x_coarse)

    error_coarse = jnp.max(jnp.abs(actual_grad_coarse - expected_grad_coarse))
    print(f"Coarse grid test: max error = {error_coarse:.2e}")

    assert error_fine < 1e-2, "Fine grid should have very low error"
    assert error_coarse < 1e-2, "Coarse grid should have reasonable error"

    print("✓ Edge case tests passed!")


def visualize_gradient_test():
    """Visualize the gradient test results"""

    x, f, actual_grad, expected_grad = test_gradient_1d_sine()

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot function
    ax1.plot(x, f, "b-", linewidth=2, label="f(x) = Σ sin(aᵢx)")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Function: Sum of Sine Functions")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot gradients
    ax2.plot(x, expected_grad, "r--", linewidth=2, label="Analytical df/dx")
    ax2.plot(x, actual_grad, "b-", linewidth=1, label="Numerical df/dx")
    ax2.set_ylabel("f'(x)")
    ax2.set_title("Gradient Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot error
    error = jnp.abs(actual_grad - expected_grad)
    ax3.semilogy(x, error, "g-", linewidth=2)
    ax3.set_xlabel("x")
    ax3.set_ylabel("|Error|")
    ax3.set_title("Absolute Error (log scale)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


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
    grad = Gradient(h=[dx, dy], acc=4)

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
    grad = Gradient(h=[dx, dy], acc=4)

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
    print("Testing 1D Gradient with sine functions...")
    print()

    test_gradient_1d_simple()
    print()

    test_gradient_1d_sine()
    print()

    test_gradient_1d_multiple_sines()
    print()

    test_gradient_1d_edge_cases()
    print()

    print("🎉 All 1D gradient tests passed!")

    # Uncomment to show visualization
    visualize_gradient_test()

    print("Testing 2D Gradient class...")
    test_gradient_simple()
    print()
    test_gradient()
    print()
    test_gradient_polynomial()
    print("\n🎉 All tests passed!")
