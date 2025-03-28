from typing import List, Dict, Set, Union
import numpy as np
from .factor import AbstractFactor


class ModelComponent:
    """
    A component of a model that contains multiple factors.

    Args:
        name (str): The name of the component.
        factors (List[AbstractFactor]): A list of factors that make up this component.

    Notes:
        - Factor names must be unique within a component.
    """
    def __init__(self, name: str, factors: List[AbstractFactor]):
        self.name = name
        self.factors = factors

        # Ensure factor names are unique
        factor_names = [factor.name for factor in factors]
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("Factor names must be unique")

    def exposed_variables(self) -> Dict[str, List[str]]:
        """
        Get the variables exposed by each factor in the model component.

        Returns:
            Dict[str, List[str]]: A dictionary mapping factor names to their exposed variables.
        """
        return {factor.name: factor.exposed_variables() for factor in self.factors}

    def required_variables(self) -> Set[str]:
        """
        Collect all variables required by any factor in the model component.

        Returns:
            Set[str]: A set containing all required variables from all factors.
        """
        return {var for factor in self.factors for var in factor.required_variables()}

    def evaluate(
        self,
        input_variables: Dict[str, Union[np.ndarray, float]],
        exposed_variables: Dict[str, Dict[str, Union[np.ndarray, float]]],
    ) -> np.ndarray:
        """
        Evaluate all factors in the model component in sequence, updating the output.

        Args:
            output (np.ndarray): Vector that will be modified by the evaluation.
            input_variables (Dict[str, Union[np.ndarray, float]]): Variables available as inputs to the factors.
            exposed_variables (Dict[str, Dict[str, Union[np.ndarray, float]]]): Variables exposed by previously evaluated factors.

        Returns:
            np.ndarray: The modified output vector.
        """

        output = 1.

        for factor in self.factors:
            output *= factor.evaluate(input_variables, exposed_variables)

        return output