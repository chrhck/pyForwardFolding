from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from .backend import backend
from .factor import AbstractFactor


class BinnedExpectation:
    """
    Represents an expectation calculated over binned data.

    Args:
        model (AbstractFactor): The model used to calculate weights for each bin.
        binning (AbstractBinning): The binning used to create the histogram.
        binned_factors (Optional[List[AbstractFactor]]): Factors to be added to the histogram.
    """
    def __init__(self, model: AbstractFactor, binning: "AbstractBinning", binned_factors: Optional[List[AbstractFactor]] = None):
        self.model = model
        self.binning = binning
        self.binned_factors = binned_factors if binned_factors is not None else []

    def required_variables(self) -> set:
        """
        Get all variables required by the BinnedExpectation.

        Returns:
            set: A union of the binning input variables and the model's required variables.
        """
        return set(self.binning.required_variables()).union(self.model.required_variables())

    def exposed_variables(self) -> List[str]:
        """
        Get variables exposed by the BinnedExpectation.

        Returns:
            List[str]: Variables exposed by the underlying model.
        """
        return self.model.exposed_variables()

    def evaluate(
        self,
        output: np.ndarray,
        weight_buffer: np.ndarray,
        weight_sq_buffer: np.ndarray,
        component_buffer: np.ndarray,
        input_variables: Dict[str, Union[np.ndarray, float]],
        exposed_variables: Dict[str, Union[np.ndarray, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a binned expectation by creating a weighted histogram.

        Args:
            output (np.ndarray): The output container. Should be initialized to zeros.
            weight_buffer (np.ndarray): Buffer for weights.
            weight_sq_buffer (np.ndarray): Buffer for squared weights.
            component_buffer (np.ndarray): Buffer for intermediate computations.
            input_variables (Dict[str, Union[np.ndarray, float]]): A collection of input variables.
            exposed_variables (Dict[str, Union[np.ndarray, float]]): Variables exposed by previously evaluated components.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The histogram weights and squared weights representing the binned expectation.
        """
        # Evaluate the model to get weights
        self.model.evaluate(weight_buffer, component_buffer, input_variables, exposed_variables)
        weight_sq_buffer[:] = backend.power(weight_buffer, 2)

        # Extract binning variables
        binning_variables = tuple(input_variables[var] for var in self.binning.required_variables())

        # Build histograms
        hist = self.binning.build_histogram(output, weight_buffer, binning_variables)
        hist_ssq = self.binning.build_histogram(output, weight_sq_buffer, binning_variables)

        # Add contributions from binned factors
        for factor in self.binned_factors:
            hist += factor.evaluate(input_variables, exposed_variables)
            # TODO: Handle uncertainty for binned factors

        # Clamp values to avoid negative or infinite values
        hist = backend.clamp(hist, 0, float("inf"))
        hist_ssq = backend.clamp(hist_ssq, 0, float("inf"))

        return hist, hist_ssq