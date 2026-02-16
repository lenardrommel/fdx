import pytest
from jax import numpy as jnp

from fdx.operators import Diff, FieldOperator, Identity, Mul, ScalarOperator
from fdx.grids import EquidistantAxis


def _interior_1d(width: int = 2) -> slice:
    return slice(width, -width)


def _interior_2d(width: int = 2) -> tuple[slice, slice]:
    return (slice(width, -width), slice(width, -width))


def _fd_atol(dx: float, acc: int, C: float = 10.0) -> float:
    """
    Calculates absolute tolerance based on grid spacing and accuracy order.
    atol ~ O(dx^p) where p is the accuracy order.
    """
    p = 2 if acc <= 2 else acc
    return C * float(dx) ** p


class TestDiffOperator:
    """Tests for the Diff operator class."""

    def test_diff_1d_polynomial(self):
        """Test simple 1st derivative of polynomial."""
        x = jnp.linspace(0.0, 2.0, 100)
        dx = x[1] - x[0]
        f = x**3
        
        # d/dx(x^3) = 3x^2
        target = 3 * x**2
        
        # Operator
        D = Diff(0, axis=EquidistantAxis(0, dx), acc=4)
        df = D(f)
        
        sl = _interior_1d(2)
        atol = _fd_atol(dx, 4, C=5.0)
        assert jnp.allclose(df[sl], target[sl], rtol=1e-3, atol=atol)

    def test_diff_high_order_via_power(self):
        """Test higher order derivative via operator power."""
        x = jnp.linspace(0.0, 2.0 * jnp.pi, 200)
        dx = x[1] - x[0]
        f = jnp.sin(x)
        
        # d^2/dx^2 sin(x) = -sin(x)
        target = -jnp.sin(x)
        
        # D^2
        D = Diff(0, axis=EquidistantAxis(0, dx), acc=4)
        D2 = D**2
        d2f = D2(f)
        
        assert D2.order == 2
        
        sl = _interior_1d(3) # higher order stencil needs more boundary
        atol = _fd_atol(dx, 4, C=10.0)
        assert jnp.allclose(d2f[sl], target[sl], rtol=1e-3, atol=atol)

    def test_diff_2d_partial(self):
        """Test partial derivatives in 2D."""
        nx, ny = 50, 50
        x = jnp.linspace(0.0, 2.0, nx)
        y = jnp.linspace(0.0, 2.0, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        
        f = X**2 + Y**3
        
        # d/dx = 2X
        Dx = Diff(0, axis=EquidistantAxis(0, dx), acc=4)
        df_dx = Dx(f)
        target_x = 2 * X
        
        # d/dy = 3Y^2
        Dy = Diff(1, axis=EquidistantAxis(1, dy), acc=4)
        df_dy = Dy(f)
        target_y = 3 * Y**2
        
        sl = _interior_2d(2)
        atol_x = _fd_atol(dx, 4)
        atol_y = _fd_atol(dy, 4)
        
        assert jnp.allclose(df_dx[sl], target_x[sl], rtol=1e-3, atol=atol_x)
        assert jnp.allclose(df_dy[sl], target_y[sl], rtol=1e-3, atol=atol_y)


class TestOperatorAlgebra:
    """Tests for algebraic combinations of operators."""

    def test_linearity_add(self):
        """Test (A + B)f = Af + Bf."""
        x = jnp.linspace(0.0, 1.0, 50)
        dx = x[1] - x[0]
        f = x**2
        
        D = Diff(0, axis=EquidistantAxis(0, dx), acc=2)
        op = D + D
        
        res = op(f)
        target = 2 * (2 * x)
        
        sl = _interior_1d(1)
        atol = _fd_atol(dx, 2, C=5.0)
        assert jnp.allclose(res[sl], target[sl], rtol=1e-3, atol=atol)

    def test_scalar_multiplication(self):
        """Test (c * A)f = c * Af."""
        x = jnp.linspace(0.0, 1.0, 50)
        dx = x[1] - x[0]
        f = x**3
        
        D = Diff(0, axis=EquidistantAxis(0, dx), acc=4)
        op = 2.5 * D
        
        res = op(f)
        target = 2.5 * (3 * x**2)
        
        sl = _interior_1d(2)
        atol = _fd_atol(dx, 4, C=5.0)
        assert jnp.allclose(res[sl], target[sl], rtol=1e-3, atol=atol)

    def test_operator_composition(self):
        """Test (A * B)f = A(B(f))."""
        x = jnp.linspace(0.0, 1.0, 50)
        dx = x[1] - x[0]
        f = x**4
        
        # D * D should equal D^2
        D = Diff(0, axis=EquidistantAxis(0, dx), acc=4)
        op = D * D
        
        res = op(f)
        target = 12 * x**2
        
        sl = _interior_1d(3)
        atol = _fd_atol(dx, 4, C=20.0)
        assert jnp.allclose(res[sl], target[sl], rtol=1e-3, atol=atol)

    def test_linear_differential_operator(self):
        """Test L = D^2 + D + 1."""
        x = jnp.linspace(0.0, 1.0, 100)
        dx = x[1] - x[0]
        f = x**3
        
        D = Diff(0, axis=EquidistantAxis(0, dx), acc=4)
        # L = D^2 + D + I
        L = (D**2) + D + Identity()
        
        res = L(f)
        target = 6*x + 3*x**2 + x**3
        
        sl = _interior_1d(3)
        atol = _fd_atol(dx, 4, C=10.0)
        assert jnp.allclose(res[sl], target[sl], rtol=1e-3, atol=atol)


class TestMatrixRepresentation:
    """Tests for matrix representations of operators."""

    def test_diff_matrix_1d(self):
        """Test that matrix @ vector matches operator(vector)."""
        nx = 20
        x = jnp.linspace(0.0, 1.0, nx)
        dx = x[1] - x[0]
        f = x**2
        
        D = Diff(0, axis=EquidistantAxis(0, dx), acc=2)
        
        # Dense matrix
        mat = D.matrix((nx,))
        assert mat.shape == (nx, nx)
        
        # Matrix-vector multiplication
        res_mat = mat @ f
        res_op = D(f)
        
        # Should be exactly equal if implementation is consistent
        # Finite difference stencils are linear operations
        assert jnp.allclose(res_mat, res_op, atol=1e-12)

    def test_composite_matrix(self):
        """Test matrix for (A + B) and Mul(A, B)."""
        nx = 15
        x = jnp.linspace(0.0, 1.0, nx)
        dx = x[1] - x[0]
        
        D = Diff(0, axis=EquidistantAxis(0, dx), acc=2)
        Id = Identity()
        
        # Op: D + I
        op_sum = D + Id
        mat_sum = op_sum.matrix((nx,))
        mat_D = D.matrix((nx,))
        mat_I = jnp.eye(nx)
        
        assert jnp.allclose(mat_sum, mat_D + mat_I)
        
        # Op: Mul(D, D) - explicitly construct Mul
        op_mul = Mul(D, D)
        mat_mul = op_mul.matrix((nx,))
        
        # Matrix multiplication of D @ D
        assert jnp.allclose(mat_mul, mat_D @ mat_D)

    def test_mixed_derivative_matrix(self):
        """Test matrix for mixed partials (D_x * D_y)."""
        nx, ny = 5, 5
        x = jnp.linspace(0.0, 1.0, nx)
        y = jnp.linspace(0.0, 1.0, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        Dx = Diff(0, axis=EquidistantAxis(0, dx), acc=2)
        Dy = Diff(1, axis=EquidistantAxis(1, dy), acc=2)
        
        # Dx * Dy -> Mul(Dx, Dy) because different dims
        op = Dx * Dy
        assert isinstance(op, Mul)
        
        mat = op.matrix((nx, ny))
        mat_x = Dx.matrix((nx, ny))
        mat_y = Dy.matrix((nx, ny))
        
        assert jnp.allclose(mat, mat_x @ mat_y)

def test_field_operator_matrix():
    """Test FieldOperator (pointwise mult) matrix."""
    nx = 10
    val = jnp.arange(nx)
    op = FieldOperator(val)
    
    mat = op.matrix((nx,))
    assert jnp.allclose(mat, jnp.diag(val))
    
    f = jnp.ones(nx)
    assert jnp.allclose(op(f), val * f)
    assert jnp.allclose(mat @ f, val * f)


def test_expression_properties_propagation():
    """Test propagation of grid and accuracy settings."""
    # Build expression without grid/acc
    D1 = Diff(0)
    D2 = Diff(0)
    op = D1 + D2
    
    # Set accuracy
    op.set_accuracy(6)
    assert D1.acc == 6
    assert D2.acc == 6
    
    # Set grid
    x = jnp.linspace(0.0, 1.0, 10)
    dx = x[1] - x[0]
    op.set_grid({0: dx})
    
    assert D1.axis.spacing == dx
    assert D2.axis.spacing == dx
    
    # Check execution
    f = x**2
    res = op(f)
    # D(x^2) + D(x^2) = 2x + 2x = 4x
    target = 4 * x
    sl = _interior_1d(3) # acc=6 needs wider boundary
    atol = _fd_atol(dx, 6, C=5.0)
    assert jnp.allclose(res[sl], target[sl], rtol=1e-3, atol=atol)
