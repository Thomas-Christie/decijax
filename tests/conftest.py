"""Shared pytest configuration for the decijax test suite.

Enabling float64 here rather than per-module guarantees the flag is set before
pytest imports any test module, since conftest is loaded first.
"""

from jax import config

config.update("jax_enable_x64", True)
