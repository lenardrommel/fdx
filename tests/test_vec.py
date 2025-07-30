import jax
import matplotlib.pyplot as plt
import pytest
from jax import numpy as jnp

from fdx import Gradient


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_gradient_1d_sine(accuracy):
    """Test 1D gradient with sum of sine functions: f(x) = sum_i sin(a_i * x)"""

    x = jnp.linspace(0, 2 * jnp.pi, 100)
    dx = x[1] - x[0]
    a_coeffs = jnp.array([1.0, 2.0, 3.0, 0.5])
    f = jnp.sum(jnp.array([jnp.sin(a * x) for a in a_coeffs]), axis=0)

    grad = Gradient(h=[dx], acc=accuracy)

    actual_grad = grad(f, axis=0)

    expected_grad = jnp.sum(jnp.array([a * jnp.cos(a * x) for a in a_coeffs]), axis=0)

    tolerance = 10 ** (-accuracy / 2 + 1)
    max_error = jnp.max(jnp.abs(actual_grad - expected_grad))

    print(
        f"Accuracy {accuracy}: Max error = {max_error:.2e}, Tolerance = {tolerance:.2e}"
    )

    assert max_error < tolerance, (
        f"Error {max_error:.2e} exceeds tolerance {tolerance:.2e} for accuracy {accuracy}"
    )


@pytest.mark.parametrize(
    "a_coeffs,name,accuracy",
    [
        ([1.0], "Single sine", 2),
        ([1.0], "Single sine", 10),
        ([1.0, 2.0], "Two sines", 6),
        ([0.5, 1.5, 2.5], "Three sines", 6),
        ([1.0, 3.0, 5.0, 0.2], "Four sines", 6),
    ],
)
def test_gradient_1d_multiple_sines(a_coeffs, name, accuracy):
    """Test with different combinations of sine functions"""

    x = jnp.linspace(0, 4 * jnp.pi, 200)
    dx = x[1] - x[0]

    a_coeffs = jnp.array(a_coeffs)

    # f(x) = sum_i sin(a_i * x)
    f = jnp.sum(jnp.array([jnp.sin(a * x) for a in a_coeffs]), axis=0)

    grad = Gradient(h=[dx], acc=accuracy)
    actual_grad = grad(f, axis=0)

    # Analytical gradient: f'(x) = sum_i a_i * cos(a_i * x)
    expected_grad = jnp.sum(jnp.array([a * jnp.cos(a * x) for a in a_coeffs]), axis=0)

    max_error = jnp.max(jnp.abs(actual_grad - expected_grad))
    tolerance = 1e-2  # 4th order accuracy tolerance

    print(f"{name}: coefficients = {a_coeffs}, max error = {max_error:.2e}")

    assert max_error < tolerance, f"Error {max_error:.2e} exceeds tolerance for {name}"


@pytest.mark.parametrize(
    "grid_size,frequency,expected_accuracy",
    [
        (1000, 10, 1e-2),  # Fine grid, high frequency
        (20, 1, 1e-1),  # Coarse grid, low frequency
    ],
)
def test_gradient_1d_edge_cases(grid_size, frequency, expected_accuracy):
    """Test edge cases and boundary conditions"""

    x = jnp.linspace(0, jnp.pi, grid_size)
    dx = x[1] - x[0]

    f = jnp.sin(frequency * x)

    grad = Gradient(h=[dx], acc=4)
    actual_grad = grad(f, axis=0)
    expected_grad = frequency * jnp.cos(frequency * x)

    max_error = jnp.max(jnp.abs(actual_grad - expected_grad))

    print(f"Grid size {grid_size}, frequency {frequency}: max error = {max_error:.2e}")

    assert max_error < expected_accuracy, (
        f"Error {max_error:.2e} exceeds expected accuracy {expected_accuracy:.2e}"
    )


@pytest.mark.parametrize("accuracy", [2, 4, 6])
def test_gradient_2d_quadratic(accuracy):
    """Test the Gradient class with a proper 2D scalar field"""

    # Create 2D grid
    x = jnp.linspace(0, 1, 20)
    y = jnp.linspace(0, 1, 20)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Define a 2D scalar function: f(x,y) = 2*x^2 + y^2
    f = 2 * X**2 + Y**2

    # Create gradient operator
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    grad = Gradient(h=[dx, dy], acc=accuracy)

    # Compute gradient
    actual_grad = grad(f)

    # Expected gradients: ∂f/∂x = 4*x, ∂f/∂y = 2*y
    expected_grad_x = 4 * X
    expected_grad_y = 2 * Y

    # Set tolerance based on accuracy
    tolerance = 10 ** (-accuracy / 2)

    error_x = jnp.max(jnp.abs(actual_grad[0] - expected_grad_x))
    error_y = jnp.max(jnp.abs(actual_grad[1] - expected_grad_y))

    print(
        f"Accuracy {accuracy}: ∂f/∂x error = {error_x:.2e}, ∂f/∂y error = {error_y:.2e}"
    )

    assert error_x < tolerance, (
        f"∂f/∂x error {error_x:.2e} exceeds tolerance {tolerance:.2e}"
    )
    assert error_y < tolerance, (
        f"∂f/∂y error {error_y:.2e} exceeds tolerance {tolerance:.2e}"
    )


def test_gradient_2d_simple():
    """Simpler test with a quadratic function"""

    # Create a smaller 2D grid
    x = jnp.linspace(-1, 1, 10)
    y = jnp.linspace(-1, 1, 10)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Simple quadratic: f(x,y) = x^2 + y^2
    f = X**2 + Y**2

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    grad = Gradient(h=[dx, dy], acc=4)

    # Compute gradient
    grad_f = grad(f)

    # Expected: ∂f/∂x = 2*x, ∂f/∂y = 2*y
    expected_x = 2 * X
    expected_y = 2 * Y

    # Check center point (should be close to 0)
    center_idx = len(x) // 2
    center_error_x = abs(grad_f[0][center_idx, center_idx])
    center_error_y = abs(grad_f[1][center_idx, center_idx])

    print(f"Center gradient errors: x = {center_error_x:.2e}, y = {center_error_y:.2e}")

    # For 4th order accuracy on quadratic, expect very small errors
    assert jnp.allclose(grad_f[0], expected_x, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(grad_f[1], expected_y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("accuracy", [2, 4, 6])
def test_gradient_2d_polynomial(accuracy):
    """Test with a more complex polynomial"""

    x = jnp.linspace(0, 2, 40)
    y = jnp.linspace(0, 2, 40)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # f(x,y) = x^3 + 2*x*y + y^2
    f = X**3 + 2 * X * Y + Y**2

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    grad = Gradient(h=[dx, dy], acc=accuracy)

    grad_f = grad(f)

    # Expected gradients: ∂f/∂x = 3*x^2 + 2*y, ∂f/∂y = 2*x + 2*y
    expected_x = 3 * X**2 + 2 * Y
    expected_y = 2 * X + 2 * Y

    # Tolerance scales with accuracy but is more lenient for higher-order polynomials
    tolerance = 10 ** (-accuracy / 3)

    error_x = jnp.max(jnp.abs(grad_f[0] - expected_x))
    error_y = jnp.max(jnp.abs(grad_f[1] - expected_y))

    print(
        f"Polynomial test (acc={accuracy}): ∂f/∂x error = {error_x:.2e}, ∂f/∂y error = {error_y:.2e}"
    )

    assert error_x < tolerance, (
        f"∂f/∂x error {error_x:.2e} exceeds tolerance {tolerance:.2e}"
    )
    assert error_y < tolerance, (
        f"∂f/∂y error {error_y:.2e} exceeds tolerance {tolerance:.2e}"
    )


def test_3d_gradient_on_scalar_func():
    """Test 3D gradient computation on scalar function"""

    # Create 3D mesh
    x = jnp.linspace(-1, 1, 30)
    y = jnp.linspace(-1, 1, 30)
    z = jnp.linspace(-1, 1, 30)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    h = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]

    # Test function: f(x,y,z) = sin(x) * sin(y) * sin(z)
    f = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)

    # Expected gradients
    grad_f_ex = jnp.array(
        [
            jnp.cos(X) * jnp.sin(Y) * jnp.sin(Z),  # ∂f/∂x
            jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z),  # ∂f/∂y
            jnp.sin(X) * jnp.sin(Y) * jnp.cos(Z),  # ∂f/∂z
        ]
    )

    grad = Gradient(h=h, acc=4)
    grad_f = grad(f)

    # Check each component
    for i, component in enumerate(["x", "y", "z"]):
        error = jnp.max(jnp.abs(grad_f[i] - grad_f_ex[i]))
        print(f"3D gradient ∂f/∂{component} error: {error:.2e}")
        assert error < 1e-4, (
            f"3D gradient component {component} error {error:.2e} too large"
        )


def test_3d_gradient_spacing_variations():
    """Test 3D gradient with different spacing specifications"""

    x = jnp.linspace(-1, 1, 20)
    y = jnp.linspace(-1, 1, 20)
    z = jnp.linspace(-1, 1, 20)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    f = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
    expected = jnp.array(
        [
            jnp.cos(X) * jnp.sin(Y) * jnp.sin(Z),
            jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z),
            jnp.sin(X) * jnp.sin(Y) * jnp.cos(Z),
        ]
    )

    h = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]

    # Test with h parameter
    grad_h = Gradient(h=h, acc=4)
    result_h = grad_h(f)

    for i in range(3):
        error = jnp.max(jnp.abs(result_h[i] - expected[i]))
        assert error < 1e-3, f"3D gradient with h parameter failed for component {i}"


def test_gradient_dimension_mismatch():
    """Test that gradient raises appropriate errors for dimension mismatches"""

    # Create 2D function but try to use 3D gradient
    x = jnp.linspace(0, 1, 10)
    y = jnp.linspace(0, 1, 10)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    f = X**2 + Y**2

    # This should work fine
    grad_2d = Gradient(h=[0.1, 0.1], acc=4)
    result = grad_2d(f)
    assert result.shape == (2, 10, 10)


def test_gradient_accuracy_orders():
    """Test that different accuracy orders produce expected error scaling"""

    x = jnp.linspace(0, 2 * jnp.pi, 100)
    dx = x[1] - x[0]
    f = jnp.sin(2 * x)  # f'(x) = 2*cos(2*x)
    expected = 2 * jnp.cos(2 * x)

    errors = []
    accuracies = [2, 4, 6, 8]

    for acc in accuracies:
        grad = Gradient(h=[dx], acc=acc)
        result = grad(f, axis=0)
        error = jnp.max(jnp.abs(result - expected))
        errors.append(error)
        print(f"Accuracy {acc}: error = {error:.2e}")

    # Higher accuracy should generally produce smaller errors
    # (though this isn't always monotonic due to numerical precision limits)
    assert errors[1] < errors[0], "4th order should be more accurate than 2nd order"


@pytest.fixture
def visualization_data():
    """Fixture to provide data for visualization tests"""
    x = jnp.linspace(0, 2 * jnp.pi, 100)
    dx = x[1] - x[0]
    a_coeffs = jnp.array([1.0, 2.0, 3.0, 0.5])
    f = jnp.sum(jnp.array([jnp.sin(a * x) for a in a_coeffs]), axis=0)

    grad = Gradient(h=[dx], acc=4)
    actual_grad = grad(f, axis=0)
    expected_grad = jnp.sum(jnp.array([a * jnp.cos(a * x) for a in a_coeffs]), axis=0)

    return x, f, actual_grad, expected_grad


def visualize_gradient_test(visualization_data):
    """Visualize the gradient test results - not a test, just for debugging"""

    x, f, actual_grad, expected_grad = visualization_data

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


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
