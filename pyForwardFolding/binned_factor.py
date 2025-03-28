from typing import Dict, Union, List
import numpy as np
from .factor import AbstractFactor


class BinnedFactor(AbstractFactor):
    """
    Represents a factor that contributes to a binned expectation.

    Args:
        name (str): The name of the factor.
        bin_variable (str): The variable used for binning.
        bin_edges (List[float]): The edges of the bins.
    """
    def __init__(self, name: str, bin_variable: str, bin_edges: List[float]):
        self.name = name
        self.bin_variable = bin_variable
        self.bin_edges = np.array(bin_edges)

    def required_variables(self) -> List[str]:
        """
        Get the variables required by this factor.

        Returns:
            List[str]: A list containing the binning variable.
        """
        return [self.bin_variable]

    def exposed_variables(self) -> List[str]:
        """
        Get the variables exposed by this factor.

        Returns:
            List[str]: An empty list since this factor does not expose variables.
        """
        return []

    def evaluate(
        self,
        output: np.ndarray,
        input_variables: Dict[str, Union[np.ndarray, float]],
        exposed_variables: Dict[str, Union[np.ndarray, float]],
    ) -> np.ndarray:
        """
        Evaluate the factor and update the output array.

        Args:
            output (np.ndarray): The output array to be updated.
            input_variables (Dict[str, Union[np.ndarray, float]]): Input variables for the factor.
            exposed_variables (Dict[str, Union[np.ndarray, float]]): Exposed variables from other factors.

        Returns:
            np.ndarray: The updated output array.
        """
        # Extract the binning variable data
        data = input_variables[self.bin_variable]

        # Initialize the output array
        output.fill(0)

        # Compute the histogram
        bin_indices = np.searchsorted(self.bin_edges, data, side="right") - 1
        for idx in bin_indices:
            if 0 <= idx < len(output):
                output[idx] += 1

        return output