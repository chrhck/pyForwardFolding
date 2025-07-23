from typing import Dict, List, Optional, Set, Tuple, Union

from .backend import ArrayType, backend
from .binning import AbstractBinning
from .factor import AbstractBinnedFactor
from .model import Model


class BinnedExpectation:
    """
    Represents an expectation calculated over binned data.

    Args:
        name (str): Name of the BinnedExpectation.
        dskey_model_pairs (List[Tuple[str, Model]]): List of tuples where each tuple contains a dataset key and a model.
        binning (AbstractBinning): The binning used to create the histogram.
        binned_factors (Optional[List[AbstractBinnedFactor]]): Factors to be added to the histogram.
        lifetime (float): Lifetime of the binned expectation.
    """
    def __init__(self,
                 name: str,
                 dskey_model_pairs: List[Tuple[str, Model]],
                 binning: AbstractBinning,
                 binned_factors: Optional[List[AbstractBinnedFactor]] = None,
                 lifetime: float = 1.0,
                ):
        self.name = name
        self.dskey_model_pairs = dskey_model_pairs
        self.models = [model for _, model in dskey_model_pairs]
        self.binning = binning
        self.binned_factors = binned_factors if binned_factors is not None else []
        self.lifetime = lifetime

    @property
    def required_variables(self) -> set:
        """
        Get all variables required by the BinnedExpectation.

        Returns:
            Set[str]: A union of the binning input variables and the model's required variables.
        """

        model_required_vars = set()
        for model in self.models:
            model_required_vars.update(model.required_variables)

        return set(self.binning.required_variables).union(model_required_vars)

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get parameters exposed by the BinnedExpectation.

        Returns:
            Dict[str, List[str]]: Variables exposed by the underlying model and binned factors.
        """

        model_exposed = set()
        
        for model in self.models:
            model_exposed.update(model.exposed_parameters)

        bf_exposed = {par for factor in self.binned_factors for par in factor.exposed_parameters}
        
        return model_exposed | bf_exposed

    def evaluate(
        self,
        datasets: Dict[str, Dict[str, Union[ArrayType, float]]],
        parameter_values: Dict[str, float],
    ) -> Tuple[ArrayType, ArrayType]:
        """
        Evaluate a binned expectation by creating a weighted histogram.

        Args:
            datasets (Dict[str, Dict[str, Union[ArrayType, float]]]): A dictionary where keys are dataset names and values are dictionaries of input variables.
            parameter_values (Dict[str, float]): A dictionary of parameter values, where keys are parameter names and values are arrays or scalars.

        Returns:
            Tuple[ArrayType, ArrayType]: A tuple containing:
                - The histogram weights (ArrayType).
                - The squared weights (ArrayType) representing the binned expectation.
        """

        hist = 0
        hist_ssq = 0

        for model_dskey, model in self.dskey_model_pairs:
            if model_dskey not in datasets:
                raise ValueError(f"Dataset '{model_dskey}' not found in provided datasets.")
            input_variables = datasets[model_dskey]

            # Evaluate the model to get weights
            weights = model.evaluate(input_variables,
                                     parameter_values,)
            weight_sq = backend.power(weights, 2)
            # Extract binning variables
            binning_variables = tuple(input_variables[var] for var in self.binning.required_variables)

            # Build histograms
            hist += self.binning.build_histogram(
                model_dskey,
                weights,
                binning_variables
                )*self.lifetime
            hist_ssq += self.binning.build_histogram(
                model_dskey,
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
