from typing import Dict, List, Set, Tuple, Union

import numpy as np

from .backend import Array
from .model_component import ModelComponent


class Model:
    """
    A forward-folding model consisting of multiple components, each with an associated baseline weight.

    Args:
        name (str): The name of the model.
        components (List[ModelComponent]): The components that make up the model.
        baseline_weights (List[str]): Symbols (as strings) representing the baseline weight for each component.

    Notes:
        - Component names must be unique within a model.
    """
    def __init__(self, name: str, components: List[ModelComponent], baseline_weights: List[str]):
        self.name = name
        self.components = components
        self.baseline_weights = baseline_weights

        # Ensure component names are unique
        component_names = [component.name for component in components]
        if len(component_names) != len(set(component_names)):
            raise ValueError("Model components must have unique names")

    @classmethod
    def from_pairs(cls, name: str, components: List[Tuple[str, ModelComponent]]) -> "Model":
        """
        Alternate constructor to create a Model from a list of (baseline_weight, component) pairs.

        Args:
            name (str): The name of the model.
            components (List[Tuple[str, ModelComponent]]): A list of (baseline_weight, component) pairs.

        Returns:
            Model: A new Model instance.
        """
        baseline_weights = [pair[0] for pair in components]
        component_list = [pair[1] for pair in components]
        return cls(name, component_list, baseline_weights)

    @property
    def required_variables(self) -> Set[str]:
        """
        Get all variables required by any component in the model.

        Returns:
            Set[str]: A set containing all required variables.
        """
        return {var for component in self.components for var in component.required_variables}

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get all parameters exposed by the model's components.

        Returns:
            Dict[str, Dict[str, str]]: A merged dictionary of all exposed parameters from all components.
        """

        return {par for component in self.components for par in component.exposed_parameters}

    @property
    def parameter_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get the mapping of model parameters to their respective components.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping component names to their parameter names.
        """
        return {component.name: component.parameter_mapping for component in self.components}

    def evaluate(
        self,
        input_variables: Dict[str, Union[Array, float]],
        parameter_values: Dict[str, float],
    ) -> Array:
        """
        Evaluate the model by computing the sum of all components.

        Args:
            input_variables (Dict[str, Union[Array, float]]): Input variables for model evaluation.
            parameter_values (Dict[str, Union[Array, float]]): Variables exposed by previously evaluated components.

        Returns:
            Array: The modified `output` containing the sum of all components.

        Raises:
            ValueError: If any baseline weight is not found in the input variables.
        """

        output = 0.
        for component, baseline_weight in zip(self.components, self.baseline_weights):
            # Get the baseline weight value
            baseline_weight_value = input_variables.get(baseline_weight, None)
            if baseline_weight_value is None:
                raise ValueError(f"Baseline weight '{baseline_weight}' not found in input variables")

            # Initialize the component buffer with the baseline weight value
            comp_eval = baseline_weight_value

            # Evaluate the component
            comp_eval *= component.evaluate(input_variables, parameter_values)

            # Accumulate the result into the output
            output += comp_eval

        return output


    def evaluate_per_component(
                    self,
        input_variables: Dict[str, Union[Array, float]],
        parameter_values: Dict[str, Union[Array, float]],
    ) -> Dict[str, Union[Array, float]]:
        """
        Evaluate each component of the model individually.
        Args:
            input_variables (Dict[str, Union[Array, float]]): Input variables for model evaluation.
            parameter_values (Dict[str, Union[Array, float]]): Variables exposed by previously evaluated components.
        Returns:
            Dict[str, Union[Array, float]]: A dictionary containing evaluation results for each component.
        Raises:
            ValueError: If any baseline weight is not found in the input variables.
        """

        # Ensure all input variables have the same length
        input_var_lengths = [len(value) if isinstance(value, Array) else 1 for value in input_variables.values()]
        if not all(length == input_var_lengths[0] for length in input_var_lengths):
            raise ValueError("All input variables must have the same length")
        output = {}
        for component, baseline_weight in zip(self.components, self.baseline_weights):
            # Get the baseline weight value
            baseline_weight_value = input_variables.get(baseline_weight, None)
            if baseline_weight_value is None:
                raise ValueError(f"Baseline weight '{baseline_weight}' not found in input variables")

            # Initialize the component buffer with the baseline weight value
            comp_eval = baseline_weight_value

            # Evaluate the component
            comp_eval *= component.evaluate(input_variables, parameter_values)

            # Accumulate the result into the output
            output[component.name] = comp_eval

        return output