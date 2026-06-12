from jax import config
import pytest

from decijax.utility_functions.base import AbstractUtilityFunctionBuilder

config.update("jax_enable_x64", True)


def test_abstract_utility_function_builder():
    with pytest.raises(TypeError):
        AbstractUtilityFunctionBuilder()
