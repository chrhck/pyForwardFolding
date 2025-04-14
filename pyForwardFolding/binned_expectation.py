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

    def exposed_parameters(self) -> List[str]:
        """
        Get variables exposed by the BinnedExpectation.

        Returns:
            List[str]: Parameters exposed by the underlying model.
        """
        return self.model.exposed_parameters()

    def evaluate(
        self,
        input_variables: Dict[str, Union[np.ndarray, float]],
        parameter_values: Dict[str, Union[np.ndarray, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a binned expectation by creating a weighted histogram.

        Args:
            input_variables (Dict[str, Union[np.ndarray, float]]): A collection of input variables.
            parameter_values (Dict[str, Union[np.ndarray, float]]): Variables exposed by previously evaluated components.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The histogram weights and squared weights representing the binned expectation.
        """
        # Evaluate the model to get weights
        weights = self.model.evaluate(input_variables, parameter_values)
        weight_sq = backend.power(weights, 2)

        # Extract binning variables
        binning_variables = tuple(input_variables[var] for var in self.binning.required_variables())

        # Build histograms
        hist = self.binning.build_histogram(weights, binning_variables)
        hist_ssq = self.binning.build_histogram(weight_sq, binning_variables)

        # Add contributions from binned factors
        for factor in self.binned_factors:
            hist += factor.evaluate(input_variables, parameter_values)
            # TODO: Handle uncertainty for binned factors

        # Clamp values to avoid negative or infinite values
        hist = backend.clip(hist, 0, float("inf"))
        hist_ssq = backend.clip(hist_ssq, 0, float("inf"))

        return hist, hist_ssq