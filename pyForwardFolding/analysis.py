from typing import Dict, Tuple, Union, Any
import numpy as np
from .binned_expectation import BinnedExpectation
from .buffers import BufferManager
from .backend import backend

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
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        exposed_variables: Dict[str, Dict[str, Union[np.ndarray, float]]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Evaluate all components in the analysis using the provided BufferManager.

        Args:
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


            # Evaluate the component
            hist, hist_ssq = comp.evaluate(
                input_vars,
                exposed_variables,
            )

            # Store results
            output_dict[comp_name] = hist
            output_ssq_dict[comp_name] = hist_ssq

        return output_dict, output_ssq_dict