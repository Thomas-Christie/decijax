"""Type aliases used throughout decijax."""

from jaxtyping import (
    Array,
    Key,
)

KeyArray = Key[Array, ""]
"""A JAX typed PRNG key, as returned by `jax.random.key`."""
