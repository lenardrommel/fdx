# FDX Test Suite

This directory contains comprehensive tests for the `fdx` library. The tests are organized in a unified structure with consistent naming and use both traditional unit tests and property-based testing.

## Test Structure

### Core Test Files

- **`conftest.py`**: Shared pytest fixtures and configuration
  - Grid fixtures (1D, 2D, 3D) for various test scenarios
  - Accuracy and derivative order parametrization
  - Helper functions for tolerance calculation

- **`test_diff_operator.py`**: Tests for the `Diff` operator
  - Basic functionality tests
  - Polynomial derivatives
  - Trigonometric functions
  - Multidimensional operations
  - Operator composition
  - Accuracy convergence
  - Property-based tests with Hypothesis

- **`test_gradient_comprehensive.py`**: Tests for the `Gradient` operator
  - 1D, 2D, and 3D gradients
  - Polynomial, trigonometric, and mixed functions
  - Edge cases and boundary conditions
  - Accuracy convergence
  - Property-based tests

- **`test_divergence_curl_laplacian.py`**: Tests for vector operators
  - Divergence in 2D and 3D
  - Curl operator (3D only)
  - Laplacian in 1D, 2D, and 3D
  - Property-based tests for each operator

- **`test_grids_comprehensive.py`**: Tests for grid functionality
  - `GridAxis`, `EquidistantAxis`, `NonEquidistantAxis`
  - Grid creation and manipulation
  - Factory functions (`make_grid`, `make_axis`)

- **`test_coefs_comprehensive.py`**: Tests for finite difference coefficients
  - Coefficient computation for various orders and accuracies
  - Uniform and non-uniform grids
  - Inverse Vandermonde matrix
  - Validation and error handling
  - Property-based symmetry tests

- **`test_findiff_compat.py`**: Tests for FinDiff compatibility layer
  - 1D, 2D, and 3D operations
  - Mixed derivatives
  - Trigonometric functions
  - Accuracy convergence

- **`test_properties_hypothesis.py`**: Advanced property-based tests
  - Linearity properties
  - Mathematical identities (power rule, chain rule, etc.)
  - Vector calculus identities
  - Extensive use of Hypothesis for generative testing

### Legacy Test Files

The following files are preserved from the original test suite:

- `test_accuracy.py`: Iterative accuracy tests
- `test_bugs.py`: Regression tests
- `test_coeffs.py`: Coefficient tests (comparison with original findiff)
- `test_findiff.py`: Basic FinDiff tests
- `test_norm.py`: Norm tests
- `test_operators.py`: Basic operator tests
- `test_speed.py`: Performance benchmarks
- `test_vec.py`: Vector operator tests
- `test_vector.py`: Additional vector tests

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Files

```bash
# Test the Diff operator
pytest tests/test_diff_operator.py

# Test gradient functionality
pytest tests/test_gradient_comprehensive.py

# Test property-based tests
pytest tests/test_properties_hypothesis.py
```

### Run Tests with Specific Markers

```bash
# Run only parametrized tests
pytest tests/ -m parametrize

# Run with verbose output
pytest tests/ -v

# Run with output from print statements
pytest tests/ -s
```

### Run Tests by Class

```bash
# Run a specific test class
pytest tests/test_diff_operator.py::TestDiffPolynomials

# Run a specific test method
pytest tests/test_diff_operator.py::TestDiffPolynomials::test_first_derivative_quadratic
```

## Test Organization

### Test Classes

Tests are organized into classes based on functionality:

- **`TestXxxBasic`**: Basic functionality and creation tests
- **`TestXxx1D/2D/3D`**: Dimension-specific tests
- **`TestXxxPolynomials`**: Tests on polynomial functions
- **`TestXxxTrigonometric`**: Tests on trigonometric functions
- **`TestXxxEdgeCases`**: Edge cases and boundary conditions
- **`TestXxxAccuracy`**: Accuracy convergence tests
- **`TestXxxPropertyBased`**: Property-based tests using Hypothesis

### Fixtures

Common fixtures are defined in `conftest.py`:

- **Grid Fixtures**:
  - `small_grid_1d`: 20 points, [0, 1]
  - `medium_grid_1d`: 100 points, [0, 2π]
  - `small_grid_2d`: 20×20 grid
  - `medium_grid_2d`: 40×40 grid
  - `small_grid_3d`: 20×20×20 grid

- **Parametrized Fixtures**:
  - `accuracy_order`: [2, 4, 6, 8]
  - `derivative_order`: [1, 2, 3, 4]

## Property-Based Testing

We use [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. These tests:

- Generate random inputs within specified ranges
- Test mathematical properties rather than specific values
- Provide better coverage than hand-written examples

### Common Properties Tested

1. **Linearity**: `D(af + bg) = aD(f) + bD(g)`
2. **Power Rule**: `d/dx(x^n) = nx^(n-1)`
3. **Trigonometric Identities**: `d/dx(sin(kx)) = k*cos(kx)`
4. **Vector Calculus Identities**: `div(curl(F)) = 0`, `Δf = div(grad(f))`

## Test Conventions

### Naming

- Test files: `test_<module>_comprehensive.py` or `test_<feature>.py`
- Test classes: `Test<Feature><Aspect>`
- Test methods: `test_<what_is_being_tested>`

### Assertions

- Use `jnp.allclose()` for numerical comparisons
- Set appropriate tolerances based on accuracy order
- Use `pytest.raises()` for exception testing

### Documentation

- Each test has a docstring explaining what is being tested
- Complex tests include inline comments
- Property-based tests state the mathematical property being tested

## Coverage

To run tests with coverage:

```bash
pytest tests/ --cov=fdx --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/`.

## Adding New Tests

When adding new tests:

1. Follow the existing structure and naming conventions
2. Add appropriate docstrings
3. Use fixtures from `conftest.py` when applicable
4. Consider adding property-based tests for mathematical properties
5. Test edge cases and error conditions
6. Ensure tests are deterministic (use fixed random seeds if needed)

## Performance Tests

Performance tests are in `test_speed.py`. To run benchmarks:

```bash
pytest tests/test_speed.py -v
```

## Dependencies

The test suite requires:

- `pytest`: Test framework
- `hypothesis`: Property-based testing
- `jax`: Core numerical operations
- `numpy`: Array operations
- `matplotlib`: Visualization (for some tests)
- `findiff`: Comparison tests with original library

## Continuous Integration

Tests should pass on all supported Python versions (3.8+) and platforms. The test suite is designed to be run in CI/CD pipelines.
