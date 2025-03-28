from typing import Dict, Tuple, Union, Any
import numpy as np
from .binned_expectation import BinnedExpectation


class Analysis:
    """
    Represents a complete analysis consisting of multiple binned components.

    Args:
        components (Dict[str, BinnedExpectation]): Dictionary mapping component names to their implementations.
    """
    def __init__(self, components: Dict[str, BinnedExpectation]):
        self.components = components

    def required_variables(self) -> set:
        """
        Get all variables required by any component in the analysis.

        Returns:
            set: A set containing all required variables.
        """
        return set.union(*(comp.required_variables() for comp in self.components.values()))

    def exposed_variables(self) -> Dict[str, Any]:
        """
        Get all variables exposed by the analysis components.

        Returns:
            Dict[str, Any]: A merged dictionary of all exposed variables from all components.
        """
        exposed = {}
        for comp in self.components.values():
            exposed.update(comp.exposed_variables())
        return exposed

    def evaluate(
        self,
        output: Dict[str, np.ndarray],
        weight_buffers: Dict[str, np.ndarray],
        weight_sq_buffers: Dict[str, np.ndarray],
        component_buffers: Dict[str, np.ndarray],
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        exposed_variables: Dict[str, Dict[str, Union[np.ndarray, float]]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Evaluate all components in the analysis.

        Args:
            output (Dict[str, np.ndarray]): Dict of output containers for each component. Will be initialized to zero.
            weight_buffers (Dict[str, np.ndarray]): Buffers for weights for each component.
            weight_sq_buffers (Dict[str, np.ndarray]): Buffers for squared weights for each component.
            component_buffers (Dict[str, np.ndarray]): Buffers for intermediate computations for each component.
            datasets (Dict[str, Dict[str, Union[np.ndarray, float]]]): A dictionary mapping component names to their input variables.
            exposed_variables (Dict[str, Dict[str, Union[np.ndarray, float]]]): Variables exposed by previously evaluated components.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: A tuple containing:
                - A dictionary mapping component names to their evaluation results (histograms).
                - A dictionary mapping component names to their squared evaluation results (histograms).
        """
        output_dict = {}
        output_ssq_dict = {}

        for comp_name, comp in self.components.items():
            # Get input variables for the component
            input_vars = datasets.get(comp_name)
            if input_vars is None:
                raise ValueError(f"No input variables found for component '{comp_name}'")

            # Get weight buffer
            weight_buffer = weight_buffers.get(comp_name)
            if weight_buffer is None:
                raise ValueError(f"No weight buffer found for component '{comp_name}'")
            weight_buffer.fill(0)

            # Get squared weight buffer
            weight_sq_buffer = weight_sq_buffers.get(comp_name)
            if weight_sq_buffer is None:
                raise ValueError(f"No weight^2 buffer found for component '{comp_name}'")
            weight_sq_buffer.fill(0)

            # Get component buffer
            component_buffer = component_buffers.get(comp_name)
            if component_buffer is None:
                raise ValueError(f"No component buffer found for component '{comp_name}'")
            component_buffer.fill(0)

            # Evaluate the component
            hist, hist_ssq = comp.evaluate(
                output[comp_name],
                weight_buffer,
                weight_sq_buffer,
                component_buffer,
                input_vars,
                exposed_variables,
            )

            # Store results
            output_dict[comp_name] = hist
            output_ssq_dict[comp_name] = hist_ssq

        return output_dict, output_ssq_dict