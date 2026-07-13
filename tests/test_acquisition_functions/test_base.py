from jax import config
import pytest

from decijax.acquisition_functions.base import AbstractAcquisitionFunctionBuilder

config.update("jax_enable_x64", True)


def test_abstract_acquisition_function_builder():
    with pytest.raises(TypeError):
        AbstractAcquisitionFunctionBuilder()
