import pytest
from decijax.acquisition_functions.base import AbstractAcquisitionFunctionBuilder


def test_abstract_acquisition_function_builder():
    with pytest.raises(TypeError):
        AbstractAcquisitionFunctionBuilder()
