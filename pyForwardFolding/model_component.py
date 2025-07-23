from typing import Dict, List, Set, Union

from .backend import Array
from .factor import AbstractUnbinnedFactor


class ModelComponent:
    """
    A component of a model that contains multiple factors.

    Args:
        name (str): The name of the component.
        factors (List[AbstractUnbinnedFactor]): A list of factors that make up this component.

    Notes:
        - Factor names must be unique within a component.
    """

    def __init__(self, name: str, factors: List[AbstractUnbinnedFactor]):
        self.name = name
        self.factors = factors

        # Ensure factor names are unique
        factor_names = [factor.name for factor in factors]
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("Factor names must be unique")

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get the variables exposed by each factor in the model component.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping factor names to their exposed variables.
        """
        exposed = set()
        for factor in self.factors:
            exposed |= set(factor.exposed_parameters)
        return exposed

    @property
    def parameter_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get the mapping of parameters to their corresponding factor names.

        Returns:
            Dict[str, str]: A dictionary mapping parameter names to their factor names.
        """
        return {factor.name: factor.parameter_mapping for factor in self.factors}

    @property
    def required_variables(self) -> Set[str]:
        """
        Collect all variables required by any factor in the model component.

        Returns:
            Set[str]: A set containing all required variables from all factors.
        """
        return {var for factor in self.factors for var in factor.required_variables}

    def evaluate(
        self,
        input_variables: Dict[str, Union[Array, float]],
        parameter_values: Dict[str, float],
    ) -> Array:
        """
        Evaluate all factors in the model component in sequence, updating the output.

        Args:
            output (Array): Vector that will be modified by the evaluation.
            input_variables (Dict[str, Union[Array, float]]): Variables available as inputs to the factors.
            parameter_values (Dict[str, Union[Array, float]]): Variables exposed by previously evaluated factors.

        Returns:
            Array: The modified output vector.
        """

        output = 1.0

        for factor in self.factors:
            output *= factor.evaluate(input_variables, parameter_values)

        return output  # type: ignore
