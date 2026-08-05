import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from decijax.maths import _erfcx


def test_erfcx_matches_scipy_across_its_whole_range():
    # We own `_erfcx`, so it needs its own coverage. These straddle the asymptotic
    # branch point (13.7 in double), the band JAX gets wrong, and the extremes where
    # the direct form would overflow or underflow.
    x = jnp.array(
        [
            1e-3,
            1.0,
            6.9,
            13.0,
            13.7 - 1e-6,
            13.7 + 1e-6,
            20.0,
            26.6,
            30.0,
            100.0,
            1e4,
            1e7,
        ]
    )
    assert jnp.allclose(_erfcx(x), scipy.special.erfcx(np.asarray(x)), rtol=1e-13)
    assert jnp.all(_erfcx(x) > 0.0)
    assert jnp.all(jnp.isfinite(jax.grad(lambda x_: jnp.sum(_erfcx(x_)))(x)))

    # Dense, since sparse points step straight over a narrow band of wrong values.
    x_dense = jnp.linspace(1e-3, 40.0, 20001)
    assert jnp.allclose(
        _erfcx(x_dense), scipy.special.erfcx(np.asarray(x_dense)), rtol=1e-13
    )


def test_jax_erfcx_is_still_broken_so_the_vendored_one_is_still_needed():
    # If this ever fails, jax-ml/jax#38607 is fixed and `_erfcx` can go.
    x = jnp.linspace(26.55, 26.64, 101)
    assert jnp.all(jax.scipy.special.erfcx(x) == 0.0)
