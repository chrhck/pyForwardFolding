from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from .backend import backend
from .factor import AbstractFactor, HistogramFactor
from .model import Model
from .binning import AbstractBinning, RectangularBinning


class BinnedExpectation:
    """
    Represents an expectation calculated over binned data.

    Args:
        model (AbstractFactor): The model used to calculate weights for each bin.
        binning (AbstractBinning): The binning used to create the histogram.
        det_config (Dict[str, Any]): Detector configuration for the binned expectation.
        binned_factors (Optional[List[AbstractFactor]]): Factors to be added to the histogram.
        lifetime (float): Lifetime of the binned expectation.
    """
    def __init__(self,
                 det_config: str,
                 model: Model,
                 binning: AbstractBinning,
                 binned_factors: Optional[Dict[str, HistogramFactor]] = None,
                 lifetime: float = 1.0,
                 excluded_comps: Optional[List[str]] = []):
        self.model = model
        self.binning = binning
        self.det_config = det_config
        self.binned_factors = binned_factors if binned_factors is not None else {}
        self.lifetime = lifetime
        self.excluded_comps = excluded_comps

    def required_variables(self) -> set:
        """
        Get all variables required by the BinnedExpectation.

        Returns:
            set: A union of the binning input variables and the model's required variables.
        """
        return set(self.binning.required_variables()).union(self.model.required_variables())

    def exposed_variables(self) -> Dict[str, List[str]]:
        """
        Get variables exposed by the BinnedExpectation.

        Returns:
            Dict[str, List[str]]: Variables exposed by the underlying model.
        """
        exposed = self.model.exposed_variables()
        for key in self.binned_factors:
            exposed.update({key: self.binned_factors[key].exposed_variables()})
        return exposed

    def evaluate(
        self,
        input_variables: Dict[str, Union[np.ndarray, float]],
        exposed_variables: Dict[str, Union[np.ndarray, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a binned expectation by creating a weighted histogram.

        Args:
            input_variables (Dict[str, Union[np.ndarray, float]]): A collection of input variables.
            exposed_variables (Dict[str, Union[np.ndarray, float]]): Variables exposed by previously evaluated components.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The histogram weights and squared weights representing the binned expectation.
        """
        # Evaluate the model to get weights
        weights = self.model.evaluate(input_variables,
                                      exposed_variables,
                                      excluded_comps=self.excluded_comps)
        weight_sq = backend.power(weights, 2)
        # Extract binning variables
        binning_variables = tuple(input_variables[var] for var in self.binning.required_variables())

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
        for key, factor in self.binned_factors.items():
            if isinstance(self.binning, RectangularBinning):
                if factor.bin_edges is None:
                    raise ValueError("Binned factors must have bin_edges defined.")
                else:
                    for j, bin_edge in enumerate(self.binning.bin_edges):
                        print("Set binning", bin_edge)
                        print("Loaded Binning", factor.bin_edges[self.det_config][j])
                        np.testing.assert_array_almost_equal(
                            bin_edge,
                            factor.bin_edges[self.det_config][j],
                            err_msg=f"Binned factor {factor.name} has different bin edges than the binning."
                        )
            hist_add, hist_ssq_add = factor.evaluate(input_variables,
                                                     exposed_variables,
                                                     self.det_config,)
            hist += hist_add*self.lifetime
            if hist_ssq_add is not None:
                hist_ssq += hist_ssq_add*self.lifetime**2

        # Clamp values to avoid negative or infinite values
        hist = backend.clip(hist, 0, float("inf"))
        hist_ssq = backend.clip(hist_ssq, 0, float("inf"))

        return hist, hist_ssq
