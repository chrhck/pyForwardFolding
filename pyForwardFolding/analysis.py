from typing import Dict, Set, Tuple, Union

from .backend import Array
from .binned_expectation import BinnedExpectation


class Analysis:
    """
    Represents a complete analysis consisting of multiple binned expectations.

    Args:
        expectations (Dict[str, BinnedExpectation]): Dictionary mapping expectation names to their objects.
    """

    def __init__(self, expectations: Dict[str, BinnedExpectation]):
        self.expectations = expectations

    @property
    def required_variables(self) -> Set[str]:
        """
        Get all variables required by any component in the analysis.

        Returns:
            Set[str]: A set containing all required variables.
        """
        return set.union(
            *(comp.required_variables for comp in self.expectations.values())
        )

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get all parameters exposed by the analysis expectations.

        Returns:
            Set[str]: A set of all exposed parameters from all expectations.
        """
        exposed = set()
        for comp in self.expectations.values():
            exposed |= comp.exposed_parameters
        return exposed

    def evaluate(
        self,
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        parameter_values: Dict[str, float],
    ) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        """
        Evaluate all expectations in the analysis.

        Args:
            datasets (Dict[str, Dict[str, Union[Array, float]]]): A dictionary mapping component names to their input variables.
            parameter_values (Dict[str, Union[Array, float]]): Variables exposed by previously evaluated expectations.

        Returns:
            Tuple[Dict[str, Array], Dict[str, Array]]: A tuple containing:
                - A dictionary mapping component names to their evaluation results (histograms).
                - A dictionary mapping component names to their squared evaluation results (histograms).
        """
        output_dict = {}
        output_ssq_dict = {}

        for comp_name, comp in self.expectations.items():
            # Evaluate the component
            hist, hist_ssq = comp.evaluate(
                datasets,
                parameter_values,
            )

            # Store results
            output_dict[comp_name] = hist
            output_ssq_dict[comp_name] = hist_ssq

        return output_dict, output_ssq_dict

    def __repr__(self):
        """
        String representation of the Analysis object.

        Returns:
            str: A string representation of the analysis.
        """
        return f"Analysis with {len(self.expectations)} expectations: {', '.join(self.expectations.keys())}"

    def __getitem__(self, item: str) -> BinnedExpectation:
        """
        Get a specific expectation by name.

        Args:
            item (str): The name of the expectation.

        Returns:
            BinnedExpectation: The corresponding expectation object.
        """
        return self.expectations[item]
