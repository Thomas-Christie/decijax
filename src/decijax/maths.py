"""A collection of numerically stable mathematical functions."""

import jax.numpy as jnp
from jax.scipy.special import erfc
from jaxtyping import (
    Array,
    Float,
)


def _log1mexp(x: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute $`\log(1 - e^x)`$ for $`x < 0`$ without catastrophic cancellation.

    Which form is accurate depends on how close $`e^x`$ is to one, hence the split
    at $`-\log 2`$. See [Mächler,
    2012](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf).

    Args:
        x: Strictly negative array.

    Returns:
        $`\log(1 - e^x)`$, elementwise.
    """
    log_2 = jnp.log(2.0)
    # Each branch is clamped to its own side of the split, so the discarded one
    # cannot poison the gradient of the surviving one.
    return jnp.where(
        x > -log_2,
        jnp.log(-jnp.expm1(jnp.maximum(x, -log_2))),
        jnp.log1p(-jnp.exp(jnp.minimum(x, -log_2))),
    )


def _erfcx(x: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute the scaled complementary error function $`e^{x^2}\text{erfc}(x)`$.

    We need to define our own erfcx function, because `jax.scipy.special.erfcx` has
    a bug where it currently returns zero on $`[26.54, 26.64]`$ (double) and
    $`[9.19, 9.42]`$ (single). Once
    [jax-ml/jax#38607](https://github.com/jax-ml/jax/issues/38607) is fixed, replace
    this function with the native JAX implementation.

    Args:
        x: Strictly positive array.

    Returns:
        $`e^{x^2}\text{erfc}(x)`$, elementwise.
    """
    # Beyond this point the asymptotic expansion below has reached machine precision.
    # It must stay *below* the point at which erfc underflows (26.54 in double, 9.19
    # in single), which is exactly the bug JAX's own implementation has.
    if x.dtype == jnp.float64:
        asymptotic_bound = 13.7
    elif x.dtype == jnp.float32:
        asymptotic_bound = 6.9
    else:
        raise NotImplementedError(f"_erfcx does not support dtype {x.dtype}.")

    # erfcx ~ (sqrt(pi) x)^-1 sum_k c_k x^-2k, with c_k = (-1)^k (2k-1)!! / 2^k, in
    # descending degree for `jnp.polyval`.
    coeffs = jnp.asarray(
        [
            7918.06640625,
            -1055.7421875,
            162.421875,
            -29.53125,
            6.5625,
            -1.875,
            0.75,
            -0.5,
            1.0,
        ],
        dtype=x.dtype,
    )

    large = x > asymptotic_bound
    # Clamped so the discarded branch can neither overflow nor poison the gradient.
    safe_x = jnp.where(large, jnp.ones_like(x), x)
    direct = jnp.exp(jnp.square(safe_x)) * erfc(safe_x)
    asymptotic = jnp.polyval(coeffs, 1.0 / jnp.square(x)) / (x * jnp.sqrt(jnp.pi))
    return jnp.where(large, asymptotic, direct)


def _log_abs_z_cdf_div_pdf(z: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute $`\log(|z| \Phi(z) / \phi(z))`$ for $`z < 0`$.

    Deep in the tail $`\Phi(z)`$ and $`\phi(z)`$ both underflow to zero even though
    their ratio is perfectly ordinary: at $`z = -40`$ both are around $`10^{-349}`$,
    while $`|z| \Phi(z) / \phi(z)`$ is $`0.999`$. `_erfcx` cancels the shared
    $`e^{-z^2/2}`$ factor analytically, so neither tiny number is ever formed.

    Args:
        z: Strictly negative array.

    Returns:
        Elementwise; strictly negative, approaching zero from below.
    """
    neg_inv_sqrt_2 = -(2.0**-0.5)
    log_sqrt_pi_div_2 = 0.5 * jnp.log(jnp.pi / 2.0)
    return jnp.log(_erfcx(neg_inv_sqrt_2 * z) * jnp.abs(z)) + log_sqrt_pi_div_2
