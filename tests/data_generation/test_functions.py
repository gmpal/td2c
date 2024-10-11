import random
import pytest
import numpy as np
from unittest.mock import patch
from typing import List

from d2c.data_generation.functions import *


@patch("random.uniform")
@patch("random.normalvariate")
def test_f_linear(mock_normalvariate, mock_uniform):
    """Test the linear function with two parents and two coefficients."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.0}

    mock_uniform.side_effect = [1.2, 1.5]
    mock_normalvariate.side_effect = [0.1]
    linear_func = f_linear(parents)
    result = linear_func(**test_inputs)

    expected = 1.2 * 0.5 + 1.5 * (-1.0) + 0.1

    assert result == expected, f"Expected {expected}, but got {result}"


@patch("random.uniform")
@patch("random.normalvariate")
def test_f_polynomial(mock_normalvariate, mock_uniform):
    """Test the polynomial function with two parents and two degrees."""
    parents = ["x", "y"]
    # degrees = [1, 3] default values
    test_inputs = {"x": 0.5, "y": -1.0}

    f_polynomial = polynomial_factory()

    # coefficients are parent/degree combinations
    # so second coefficient is parent 1 degree 3
    mock_uniform.side_effect = [1.2, 1.5, 1.8, 2.1]
    mock_normalvariate.side_effect = [0.1]
    poly_func = f_polynomial(parents)
    result = poly_func(**test_inputs)

    expected = (
        1.2 * (0.5**1) + 1.8 * (-(1.0**1)) + 1.5 * ((0.5**3)) + 2.1 * (-(1.0**3))
    ) + 00.1

    assert result == expected, f"Expected {expected}, but got {result}"


@patch("random.uniform")
@patch("random.normalvariate")
def test_f_sigmoid(mock_normalvariate, mock_uniform):
    """Test the sigmoid function with two parents."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.0}

    mock_uniform.side_effect = [1.2, 1.5, 0.5]
    mock_normalvariate.side_effect = [0.1]

    sigmoid_func = f_sigmoid(parents)
    result = sigmoid_func(**test_inputs)

    expected = 1 / (1 + np.exp(-(0.5 + 1.2 * 0.5 + 1.5 * (-1.0)))) + 0.1

    assert result == expected, f"Expected {expected}, but got {result}"


@patch("random.uniform")
@patch("random.normalvariate")
def test_f_nonlinear(mock_normalvariate, mock_uniform):
    """Test the nonlinear function with two parents and a custom nonlinearity."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.0}
    f_nonlinear = nonlinear_factory()

    mock_uniform.side_effect = [1.2, 1.5]
    mock_normalvariate.side_effect = [0.1]
    sin_func = f_nonlinear(parents)
    result = sin_func(**test_inputs)

    expected = np.sin(1.2 * 0.5 + 1.5 * (-1.0)) + 0.1

    assert result == pytest.approx(
        expected, rel=1e-7
    ), f"Expected {expected}, but got {result}"


@patch("random.uniform")
@patch("random.normalvariate")
def test_f_nonlinear_custom(mock_normalvariate, mock_uniform):
    """Test the nonlinear function with two parents and a custom nonlinearity."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.0}

    def custom_nonlinearity(x):
        return x**2

    f_nonlinear = nonlinear_factory(custom_nonlinearity)

    mock_uniform.side_effect = [1.2, 1.5]
    mock_normalvariate.side_effect = [0.1]

    custom_func = f_nonlinear(parents)
    result = custom_func(**test_inputs)

    expected = custom_nonlinearity(1.2 * 0.5 + 1.5 * (-1.0)) + 0.1

    assert result == expected, f"Expected {expected}, but got {result}"


@patch("random.uniform")
@patch("random.normalvariate")
def test_f_interaction(mock_normalvariate, mock_uniform):
    """Test the interaction function with two parents."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.2}

    mock_uniform.side_effect = [1.2, 1.5, 1.6]
    mock_normalvariate.side_effect = [0.1]

    interaction_func = f_interaction(parents)
    result = interaction_func(**test_inputs)

    expected = 1.2 * 0.5 + 1.5 * (-1.2) + 1.6 * 0.5 * (-1.2) + 0.1

    assert result == expected, f"Expected {expected}, but got {result}"
