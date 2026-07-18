import pytest
from decijax.acquisition_functions.base import AbstractAcquisitionFunctionBuilder
from jax import config

config.update("jax_enable_x64", True)


def test_abstract_acquisition_function_builder():
    with pytest.raises(TypeError):
        AbstractAcquisitionFunctionBuilder()
