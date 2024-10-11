import random
from typing import List
import math

__all__ = [
    "f_linear",
    "polynomial_factory",
    "f_sigmoid",
    "nonlinear_factory",
    "f_interaction",
]


def f_linear(parents: List[str]):
    """
    Usage:
        >>> linear_func = f_linear(['x', 'y'])
        >>> result = linear_func(x=1.0, y=2.0)
    """
    # Generate random weights for each parent variable
    weights = {p: random.uniform(0.5, 2.0) for p in parents}

    # Default value if no kwargs are provided
    default_value = 0.0

    # Define the function f that takes keyword arguments (**kwargs)
    def f(**kwargs):
        # If no keyword arguments are provided, use the default value
        if len(kwargs) == 0:
            mu = default_value
        else:
            mu = 0.0

        # Calculate the linear combination of weights and values provided in kwargs
        for p in parents:
            mu += weights[p] * kwargs[p]

        # Return the result of the linear combination
        return mu + random.normalvariate(0, 0.1)  # TODO: make the noise adaptable

    # Return the function f itself
    return f


def polynomial_factory(degrees: List[int] = None):
    """
    Creates a polynomial function with fixed degrees.
    """
    # avoid mutable default arguments
    if degrees is None:
        degrees = [1, 3]  # Default degrees

    def f_polynomial(parents: List[str]):
        """
        Explanation:
            - Degrees: You specify which degrees to include in the polynomial.
            - Coefficients: Random coefficients are generated for each combination of parent variable and degree.
            - Computation: The function sums the terms $a_{p, d} \cdot x_p^d$ for all $p$ and $d$.

        Usage:
            >>> poly_func = f_polynomial(['x', 'y'], degrees=[1, 3])
            >>> result = poly_func(x=2.0, y=1.5)

        """

        # Generate random coefficients for each term in the polynomial
        coefficients = {
            (p, d): random.uniform(0.5, 2.0) for p in parents for d in degrees
        }
        default_value = 0.0

        def f(**kwargs):
            mu = default_value
            for p in parents:
                for d in degrees:
                    value = kwargs.get(p, 0.0)
                    mu += coefficients[(p, d)] * (value**d)
            return mu + random.normalvariate(0, 0.1)  # TODO: make the noise adaptable

        return f

    return f_polynomial


def f_sigmoid(parents: List[str]):
    """A sigmoid function introduces nonlinearity and is often used in neural networks.

    Usage:
        >>> sigmoid_func = f_sigmoid(['x', 'y'])
        >>> result = sigmoid_func(x=0.5, y=-1.0)
    """

    # Generate random weights and bias
    weights = {p: random.uniform(-2.0, 2.0) for p in parents}
    bias = random.uniform(-1.0, 1.0)

    def f(**kwargs):
        z = bias
        for p in parents:
            value = kwargs.get(p, 0.0)
            z += weights[p] * value
        mu = 1 / (1 + math.exp(-z))
        return mu + random.normalvariate(0, 0.1)  # TODO: make the noise adaptable

    return f


def nonlinear_factory(nonlinearity: callable = math.sin):
    """
    Creates a nonlinear function with a custom nonlinearity.
    """

    def f_nonlinear(parents: List[str]):
        """
        You can create functions with custom nonlinear relationships.

        Usage:
            >>> sin_func = f_nonlinear(['x', 'y'], math.sin)
            >>> result = sin_func(x=1.0, y=2.0)
        """

        # Generate random weights
        weights = {p: random.uniform(0.5, 2.0) for p in parents}
        default_value = 0.0

        def f(**kwargs):
            mu = default_value
            for p in parents:
                value = kwargs.get(p, 0.0)
                mu += weights[p] * value
            mu = nonlinearity(mu)
            return mu + random.normalvariate(0, 0.1)  # TODO: make the noise adaptable

        return f

    return f_nonlinear


def f_interaction(parents: List[str]):
    """

    Explanation:
        - Interaction Weights: Random weights for each pair of variables.
        - Computation: w1p1 + w2p2 + w3p1p2
        Includes terms like $w_{p 1, p 2} x_{p 1} x_{p 2}$.

    Usage:
        >>> interaction_func = f_interaction(['x', 'y', 'z'])
        >>> result = interaction_func(x=1.0, y=2.0, z=3.0)
    """

    # Generate random weights for individual and interaction terms
    weights = {p: random.uniform(0.5, 2.0) for p in parents}
    interaction_weights = {
        (p1, p2): random.uniform(0.5, 2.0)
        for i, p1 in enumerate(parents)
        for p2 in parents[i + 1 :]
    }

    def f(**kwargs):
        mu = 0.0
        for p in parents:
            value = kwargs.get(p, 0.0)
            mu += weights[p] * value

        for (p1, p2), w in interaction_weights.items():
            val1 = kwargs.get(p1, 0.0)
            val2 = kwargs.get(p2, 0.0)
            mu += w * val1 * val2

        return mu + random.normalvariate(0, 0.1)  # TODO: make the noise adaptable

    return f
