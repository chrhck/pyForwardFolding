from typing import List, Dict, Set, Union, Tuple
import numpy as np
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

    def required_variables(self) -> Set[str]:
        """
        Get all variables required by any component in the model.

        Returns:
            Set[str]: A set containing all required variables.
        """
        return {var for component in self.components for var in component.required_variables()}

    def exposed_variables(self) -> Dict[str, List[str]]:
        """
        Get all variables exposed by the model's components.

        Returns:
            Dict[str, List[str]]: A merged dictionary of all exposed variables from all components.
        """
        exposed = {}
        for component in self.components:
            exposed.update(component.exposed_variables())
        return exposed

    def evaluate(
        self,
        output: np.ndarray,
        component_buffer: np.ndarray,
        input_variables: Dict[str, Union[np.ndarray, float]],
        exposed_variables: Dict[str, Dict[str, Union[np.ndarray, float]]],
    ) -> np.ndarray:
        """
        Evaluate the model by computing the sum of all components.

        Args:
            output (np.ndarray): The output array to accumulate results in. Should be initialized to zeros.
            component_buffer (np.ndarray): Buffer used for component evaluations to avoid allocations.
            input_variables (Dict[str, Union[np.ndarray, float]]): Input variables for model evaluation.
            exposed_variables (Dict[str, Dict[str, Union[np.ndarray, float]]]): Variables exposed by previously evaluated components.

        Returns:
            np.ndarray: The modified `output` containing the sum of all components.

        Raises:
            ValueError: If any baseline weight is not found in the input variables.
        """
        if len(component_buffer) != len(output):
            raise ValueError("Component buffer must be as long as the output")

        # Ensure all input variables have the same length
        input_var_lengths = [len(value) if isinstance(value, np.ndarray) else 1 for value in input_variables.values()]
        if not all(length == input_var_lengths[0] for length in input_var_lengths):
            raise ValueError("All input variables must have the same length")

        for component, baseline_weight in zip(self.components, self.baseline_weights):
            # Get the baseline weight value
            baseline_weight_value = input_variables.get(baseline_weight, None)
            if baseline_weight_value is None:
                raise ValueError(f"Baseline weight '{baseline_weight}' not found in input variables")

            # Initialize the component buffer with the baseline weight value
            component_buffer[:] = baseline_weight_value

            # Evaluate the component
            component.evaluate(component_buffer, input_variables, exposed_variables)

            # Accumulate the result into the output
            output += component_buffer

        return output