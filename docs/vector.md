# Vector calculus operators (finite differences)

Vector calculus operators built from scalar finite differences via `FinDiff`. This module provides **operator objects** you instantiate once (with grid information + accuracy), then **call** on arrays to compute derivatives.

## Design goals

- **Uniform grids**: pass `h=[dx, dy, ...]`.
- **Non-uniform grids**: pass `coords=[x, y, ...]` as 1D coordinate arrays.
- Operators are **composable**: e.g. `Laplacian(h)(f)` or `Divergence(h)(Gradient(h)(f))` (if you align shapes).
- Most operators support **N dimensions**, with **Curl** restricted to 3D.

---

## Common concepts

### Grid specification

You must specify exactly one of:

- `h`: list/array of grid spacings for a uniform grid
- `coords`: list of coordinate arrays for each axis for non-uniform grids

### Accuracy

Most operators accept:

- `acc` (default `2`): accuracy order passed through to `FinDiff`.

### Array type

All operators expect JAX arrays (`jax.Array` / `jnp.ndarray`). Your code uses a project alias:

- `fdx.types.Array`

---

## `VectorOperator`

Base class for all vector differential operators.

!!! note "Do not instantiate directly"
    Use child classes for specific operators.

### Parameters

- `acc: int` (default `2`)
- `h: list[float] | jax.Array`
- `coords: list[Array]`

### Attributes

- `acc: int` – accuracy order
- `h: Array` – spacings (uniform grids only)
- `ndims: int` – number of spatial dimensions
- `components: list[FinDiff]` – one first-derivative `FinDiff` per axis

---

## `Gradient`

N-dimensional gradient operator.

Mathematically:

\[
\nabla f = \left(\partial_{x_0} f, \partial_{x_1} f, \dots, \partial_{x_{N-1}} f\right)
\]

### Init

```python
Gradient(*, h=None, coords=None, acc=2, **kwargs)
```

Exactly one of `h` or `coords` must be provided.

### Call

```python
grad = Gradient(h=[dx, dy], acc=2)

out = grad(f)                       # full gradient
out_axis = grad(f, axis=0)          # only ∂f/∂x0
out_batched = grad(fb, has_batch=True)
```

#### Parameters

* `f: Array`
    * Scalar field.
* `axis: int | None` (default `None`)
    * If `None`, returns all partials (full gradient).
    * If `int`, returns derivative along that spatial axis.
* `has_batch: bool` (default `False`)
    * If `True`, the first axis of `f` is treated as batch.

#### Shapes

Let `ndims = N`, spatial shape `S = (s0, s1, ..., s{N-1})`.

* **No batch**
    * input: `f.shape == S`
    * output (full): `(N, *S)`
    * output (axis): `(*S)`
* **With batch**
    * input: `f.shape == (B, *S)`
    * output (full): `(B, N, *S)`
    * output (axis): `(B, *S)`

---

## `Divergence`

N-dimensional divergence operator (vector field → scalar field).

\[
\mathrm{div}(\mathbf{v}) = \nabla \cdot \mathbf{v}
\]

### Init

```python
Divergence(*, h=None, coords=None, acc=2, **kwargs)
```

### Call

```python
div = Divergence(h=[dx, dy])
out = div(v)
```

#### Parameters

* `f: Array`
    * Vector field, stored as `f[axis]` being the component along that axis.

#### Expected shape

* `f.shape == (ndims, *spatial)`

#### Output shape

* `(*spatial)`

!!! info "Implementation Note"
    The current implementation checks shape using:
    `if len(f.shape) != ndims + 1 and f.shape[0] != ndims: ...`
    which means it will only error when *both* are wrong. If you want stricter checking, change `and` → `or`.

---

## `Curl`

Curl operator for 3D vector fields only.

\[
\nabla \times \mathbf{v}
\]

### Init

```python
Curl(*, h=None, coords=None, acc=2, **kwargs)
```

Raises `ValueError` unless `ndims == 3`.

### Call

```python
curl = Curl(h=[dx, dy, dz])
out = curl(v)
```

#### Parameters

* `f: Array`
    * Vector field with components stored in axis 0.

#### Shapes

* input: `(3, nx, ny, nz)`
* output: `(3, nx, ny, nz)`

---

## `Laplacian`

N-dimensional Laplacian for scalar fields.

\[
\Delta f = \sum_{k=0}^{N-1} \partial^2_{x_k} f
\]

### Init

```python
Laplacian(h: list[float] | None = None, acc: int = 2)
```

* `h` defaults to `[1.0]` if not provided.

### Call

```python
lap = Laplacian(h=[dx, dy])
out = lap(f)
```

#### Parameters

* `f: Array` – scalar field

#### Shapes

* input: `(*spatial)`
* output: `(*spatial)`

#### Notes

This class internally creates second-derivative `FinDiff` parts and sums them.

---

## `Jacobian`

Jacobian of a **vector-valued** field with respect to spatial variables.

This supports arbitrary component shapes (e.g. multiple channels, tensors), by flattening and reshaping.

### Init

```python
Jacobian(*, h=None, coords=None, acc=2, **kwargs)
```

### Call

```python
J = Jacobian(h=[dx, dy])

out = J(u)                     # no batch
out_b = J(ub, has_batch=True)  # batch
```

#### Parameters

* `u: Array`
    * No batch: `u.shape = (*spatial, *components)`
    * With batch: `u.shape = (batch, *spatial, *components)`
* `has_batch: bool` (default `False`)
    * Whether the first axis is a batch axis.

#### Shapes

Let `spatial = (*S)` with `len(S) == ndims`, and `components = (*C)`.

* **No batch**
    * input: `(*S, *C)`
    * output: `(ndims, *S, *C)`
* **With batch**
    * input: `(B, *S, *C)`
    * output: `(B, ndims, *S, *C)`

---

## `wrap_in_ndarray`

Utility to convert scalars / lists into a JAX array.

```python
wrap_in_ndarray(value: Array | list[float]) -> Array
```

* If `value` is scalar-like: returns `jnp.array([value])`
* If `value` is list/array-like: returns `jnp.array(value)` preserving shape

---

## Examples

### 2D gradient + divergence

```python
import jax.numpy as jnp
from fdx.vector import Gradient, Divergence

nx, ny = 64, 64
dx, dy = 0.1, 0.2

x = jnp.linspace(0, (nx - 1) * dx, nx)
y = jnp.linspace(0, (ny - 1) * dy, ny)
X, Y = jnp.meshgrid(x, y, indexing="ij")

f = jnp.sin(X) * jnp.cos(Y)

grad = Gradient(h=[dx, dy], acc=2)
div = Divergence(h=[dx, dy], acc=2)

g = grad(f)        # (2, nx, ny)
d = div(g)         # (nx, ny)  (div of grad is Laplacian if consistent)
```

### Batched gradient

```python
B, nx = 8, 128
dx = 0.05
f = jnp.ones((B, nx))

grad = Gradient(h=[dx], acc=2)
dfdx = grad(f, axis=0, has_batch=True)  # (B, nx)
```

---

## Extending the API (planned / future arguments)

* **Boundary handling** (e.g. `mode="periodic" | "one-sided" | "mirror"`)
* **Stencil control** (explicit stencil or scheme selection)
* **Axis conventions** (support vector fields stored in last axis vs first axis)
* **dtype / precision** (force `float32` vs `float64`)
* **jit/vmap helpers** (flags or convenience methods for compilation strategies)
* **batch axis position** (generalize beyond “batch is axis 0”)

---

## See also

* `FinDiff` (scalar finite difference operator used internally)
* JAX: `jax.vmap`, `jax.jit`
