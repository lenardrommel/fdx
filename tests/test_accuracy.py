# test_accuracy.py

import jax.numpy as jnp
from jax import grad

from fdx import Diff


def test_iterative_accuracy():
    ns = jnp.logspace(2, 3, 10)

    def compute_errs(acc):
        errs_a = []
        errs_b = []
        errs_c = []
        errs_d = []

        for n in ns:
            x = jnp.linspace(0.1, 1, int(n))
            dx = x[1] - x[0]

            D = Diff(0, dx, acc=acc)
            D2 = D**2
            DD = D * D

            f = x**8
            func_f = lambda x: x**8
            df = grad(func_f)
            ddf = grad(df)

            exact = 8 * 7 * x**6
            actual_a = D2(f)
            actual_b = D(D(f))
            actual_c = DD(f)
            actual_d = jnp.vectorize(ddf)(x)

            err_a = jnp.max(jnp.abs((actual_a - exact) / exact))
            err_b = jnp.max(jnp.abs((actual_b - exact) / exact))
            err_c = jnp.max(jnp.abs((actual_c - exact) / exact))
            err_d = jnp.max(jnp.abs((actual_d - exact + 1e-9) / exact))
            errs_a.append(err_a)
            errs_b.append(err_b)
            errs_c.append(err_c)
            errs_d.append(err_d)

        slope_a = jnp.abs(loglog_slope(ns, jnp.array(errs_a)))
        slope_b = jnp.abs(loglog_slope(ns, jnp.array(errs_b)))
        slope_c = jnp.abs(loglog_slope(ns, jnp.array(errs_c)))
        return slope_a, slope_b, slope_c

    slope_a, slope_b, slope_c = compute_errs(2)

    assert jnp.abs(slope_a - 2) < 0.2
    # applying operators iteratively should reduce the order by one at a time:
    assert jnp.abs(slope_b - 1) < 0.2
    assert jnp.abs(slope_c - 2) < 0.2

    slope_a, slope_b, slope_c = compute_errs(4)
    assert jnp.abs(slope_a - 4) < 0.2
    # applying operators iteratively should reduce the order by one at a time:
    assert jnp.abs(slope_b - 3) < 0.4
    assert jnp.abs(slope_c - 4) < 0.2


def loglog_slope(x, y):
    slope, intercept = jnp.polyfit(jnp.log(x), jnp.log(y), 1)
    return slope
