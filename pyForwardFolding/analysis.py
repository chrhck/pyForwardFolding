from typing import Any, Dict, Tuple, Union

import numpy as np

from .binned_expectation import BinnedExpectation


class Analysis:
    """
    Represents a complete analysis consisting of multiple binned expectations.

    Args:
        expectations (Dict[str, BinnedExpectation]): Dictionary mapping expectation names to their objects.
    """
    def __init__(self, expectations: Dict[str, BinnedExpectation]):
        self.expectations = expectations

    def required_variables(self) -> set:
        """
        Get all variables required by any component in the analysis.

        Returns:
            set: A set containing all required variables.
        """
        return set.union(*(comp.required_variables() for comp in self.expectations.values()))

    def exposed_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters exposed by the analysis expectations.

        Returns:
            Dict[str, Any]: A merged dictionary of all exposed variables from all expectations.
        """
        exposed = {}
        for comp in self.expectations.values():
            exposed.update(comp.exposed_parameters())
        return exposed

    def evaluate(
        self,
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        parameter_values: Dict[str, Union[np.ndarray, float]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Evaluate all expectations in the analysis using the provided BufferManager.

        Args:
            datasets (Dict[str, Dict[str, Union[np.ndarray, float]]]): A dictionary mapping component names to their input variables.
            parameter_values (Dict[str, Union[np.ndarray, float]]): Variables exposed by previously evaluated expectations.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: A tuple containing:
                - A dictionary mapping component names to their evaluation results (histograms).
                - A dictionary mapping component names to their squared evaluation results (histograms).
        """
        output_dict = {}
        output_ssq_dict = {}

        for comp_name, comp in self.expectations.items():
            # Get input variables for the component
            input_vars = datasets.get(comp_name)
            if input_vars is None:
                raise ValueError(f"No input variables found for component '{comp_name}'")

            # Evaluate the component
            hist, hist_ssq = comp.evaluate(
                input_vars,
                parameter_values,
            )

            # Store results
            output_dict[comp_name] = hist
            output_ssq_dict[comp_name] = hist_ssq

        return output_dict, output_ssq_dict
