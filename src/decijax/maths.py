"""A collection of numerically stable mathematical functions."""

import jax.numpy as jnp
from jax.scipy.special import erfcx
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


def _log_abs_z_cdf_div_pdf(z: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute $`\log(|z| \Phi(z) / \phi(z))`$ for $`z < 0`$.

    Deep in the tail $`\Phi(z)`$ and $`\phi(z)`$ both underflow to zero even though
    their ratio is perfectly ordinary: at $`z = -40`$ both are around $`10^{-349}`$,
    while $`|z| \Phi(z) / \phi(z)`$ is $`0.999`$. `erfcx` cancels the shared
    $`e^{-z^2/2}`$ factor analytically, so neither tiny number is ever formed.

    Args:
        z: Strictly negative array.

    Returns:
        Elementwise; strictly negative, approaching zero from below.
    """
    neg_inv_sqrt_2 = -(2.0**-0.5)
    log_sqrt_pi_div_2 = 0.5 * jnp.log(jnp.pi / 2.0)
    return jnp.log(erfcx(neg_inv_sqrt_2 * z) * jnp.abs(z)) + log_sqrt_pi_div_2
