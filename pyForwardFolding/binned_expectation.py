from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .backend import backend
from .binning import AbstractBinning, RectangularBinning
from .factor import AbstractBinnedFactor
from .model import Model


class BinnedExpectation:
    """
    Represents an expectation calculated over binned data.

    Args:
        name (str): Name of the BinnedExpectation.
        model (AbstractFactor): The model used to calculate weights for each bin.
        binning (AbstractBinning): The binning used to create the histogram.
        binned_factors (Optional[List[AbstractFactor]]): Factors to be added to the histogram.
        lifetime (float): Lifetime of the binned expectation.
    """
    def __init__(self,
                 name: str,
                 model: Model,
                 binning: AbstractBinning,
                 binned_factors: Optional[List[AbstractBinnedFactor]] = None,
                 lifetime: float = 1.0,
                ):
        self.name = name
        self.model = model
        self.binning = binning
        self.binned_factors = binned_factors if binned_factors is not None else []
        self.lifetime = lifetime

    @property
    def required_variables(self) -> set:
        """
        Get all variables required by the BinnedExpectation.

        Returns:
            set: A union of the binning input variables and the model's required variables.
        """
        return set(self.binning.required_variables).union(self.model.required_variables)

    @property
    def exposed_parameters(self) -> Dict[str, List[str]]:
        """
        Get parameters exposed by the BinnedExpectation.

        Returns:
            Dict[str, List[str]]: Variables exposed by the underlying model.
        """
        model_exposed = self.model.exposed_parameters
        bf_exposed = {par for factor in self.binned_factors for par in factor.exposed_parameters}
        
        return model_exposed | bf_exposed

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
        weights = self.model.evaluate(input_variables,
                                      parameter_values,)
        weight_sq = backend.power(weights, 2)
        # Extract binning variables
        binning_variables = tuple(input_variables[var] for var in self.binning.required_variables)

        # Build histograms
        hist = self.binning.build_histogram(
            weights,
            binning_variables
            )*self.lifetime
        hist_ssq = self.binning.build_histogram(
            weight_sq,
            binning_variables
            )*self.lifetime**2

        # Add contributions from binned factors
        for factor in self.binned_factors:
            # TODO: Update logic. Do we want to store the bin edges in the factor?
            
            # if isinstance(self.binning, RectangularBinning):
            #     if factor.bin_edges is None:
            #         raise ValueError("Binned factors must have bin_edges defined.")
            #     else:
            #         for j, bin_edge in enumerate(self.binning.bin_edges):
            #             print("Set binning", bin_edge)
            #             print("Loaded Binning", factor.bin_edges[self.det_config][j])
            #             np.testing.assert_array_almost_equal(
            #                 bin_edge,
            #                 factor.bin_edges[self.det_config][j],
            #                 err_msg=f"Binned factor {factor.name} has different bin edges than the binning."
            #             )
            hist_add, hist_ssq_add = factor.evaluate(input_variables,
                                                     parameter_values,
                                                    )
            hist += hist_add*self.lifetime
            if hist_ssq_add is not None:
                hist_ssq += hist_ssq_add*self.lifetime**2

        # Clamp values to avoid negative or infinite values
        hist = backend.clip(hist, 0, float("inf"))
        hist_ssq = backend.clip(hist_ssq, 0, float("inf"))

        return hist, hist_ssq
