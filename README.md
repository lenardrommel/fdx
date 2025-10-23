<div align="center">
  <img src="fdx.png" alt="fdx logo" width="180" />
</div>

# fdx — Finite Differences in JAX

fdx is a JAX-first reimplementation of key features from the excellent findiff package — providing finite-difference derivatives, vector calculus operators, and matrix/stencil representations, all compatible with JAX arrays and transformations.

The project aims to be small, predictable, and well-tested, while keeping the essential capabilities most users need.

## Highlights

- JAX-compatible: works with `jax.numpy` arrays and APIs
- 1D/ND derivatives with controllable accuracy (even orders)
- Uniform and non-uniform grids, optional periodicity
- Vector operators: Gradient, Divergence, Curl, Laplacian
- Stencil and matrix representations for operators
- Double precision defaults for stable high-order stencils

If you need the full feature set (PDE solvers, boundary handling helpers, rich stencils), consider the original findiff.

## Install

fdx targets Python 3.8+ and requires JAX.

```bash
pip install jax  # or jax[cuda] for GPU
pip install -e .
```

## Quickstart

Compute first and second derivatives on a uniform grid:

```python
import jax.numpy as jnp
from fdx import Diff

x = jnp.linspace(0, 2*jnp.pi, 100)
dx = x[1] - x[0]
f = jnp.sin(x)

d_dx = Diff(0, grid=dx, acc=4)
df = d_dx(f)                 # ≈ cos(x)

d2_dx2 = d_dx ** 2           # second derivative along axis 0
d2f = d2_dx2(f)              # ≈ -sin(x)
```

Set periodicity or pass coordinates to use non-uniform grids:

```python
coords = jnp.cumsum(jnp.array([0.1, 0.12, 0.08, 0.11, 0.09]))
d_dx_nonuni = Diff(0, grid=coords, acc=4)

d_dx_periodic = Diff(0, grid=dx, periodic=True, acc=6)
```

## Vector Operators

```python
import jax.numpy as jnp
from fdx import Gradient, Laplacian

x = jnp.linspace(0, 2*jnp.pi, 100)
y = jnp.linspace(0, 2*jnp.pi, 120)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = jnp.meshgrid(x, y, indexing='ij')
f = jnp.sin(X) * jnp.cos(Y)

grad = Gradient(h=[dx, dy], acc=6)
gx, gy = grad(f)  # ∂f/∂x, ∂f/∂y

lap = Laplacian(h=[dx, dy], acc=4)
lf = lap(f)       # ∂²f/∂x² + ∂²f/∂y²
```

## Examples

Open the notebooks in `examples/` for runnable walkthroughs:

- examples/examples-basic.ipynb — Quick intro (ported from findiff’s basic examples)
- examples/examples-derivatives-1d.ipynb — 1D first/second derivatives and accuracy
- examples/examples-2d-operators.ipynb — Gradient and Laplacian in 2D

## Accuracy and Precision

Stencil coefficients are computed in float64 by default for stability with higher orders and tight tolerances. You can switch at runtime if needed:

```python
from fdx import set_dtype
import jax.numpy as jnp

set_dtype(jnp.float32)  # use with care; may reduce accuracy
```

## API Overview

- `Diff(dim, grid=None, periodic=False, acc=2)` — 1D/ND partial derivative along `dim`.
  - `grid` can be a spacing (float) or coordinate array; coordinates imply a non-uniform axis.
  - Use `d ** n` or `d * d` (same axis) for higher-order derivatives.
- `Gradient(h=[...], acc=...)`, `Divergence(...)`, `Curl(...)`, `Laplacian(...)` — vector calculus operators for uniform grids.
- `coefficients(...)` — raw finite-difference coefficients.
- `.matrix(shape)` and `.stencil(shape)` — operator representations for a given grid shape.

## Testing

Run the project’s tests:

```bash
pytest -q
```

## Acknowledgments

This work draws inspiration from and thanks the authors of the original [findiff](https://github.com/maroba/findiff) project.
