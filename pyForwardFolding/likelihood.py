from typing import Dict, Union, Tuple
import numpy as np
from .analysis import Analysis


class AbstractLikelihood:
    """
    Abstract base class representing a likelihood function to be evaluated against observed data.
    """
    def get_analysis(self) -> Analysis:
        raise NotImplementedError

    def llh(
        self,
        outputs: Dict[str, np.ndarray],
        observed_data: Dict[str, np.ndarray],
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        exposed_variables: Dict[str, Dict[str, Union[np.ndarray, float]]],
        empty_bins: str = "skip",
    ) -> float:
        raise NotImplementedError


class PoissonLikelihood(AbstractLikelihood):
    """
    A likelihood function that assumes Poisson-distributed data.

    Args:
        analysis (Analysis): The analysis to evaluate against observed data.
    """
    def __init__(self, analysis: Analysis):
        self.analysis = analysis

    def get_analysis(self) -> Analysis:
        """
        Get the analysis associated with this likelihood.

        Returns:
            Analysis: The associated analysis.
        """
        return self.analysis

    def llh(
        self,
        outputs: Dict[str, np.ndarray],
        observed_data: Dict[str, np.ndarray],
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        exposed_variables: Dict[str, Dict[str, Union[np.ndarray, float]]],
        empty_bins: str = "skip",
    ) -> float:
        """
        Compute the log-likelihood between model predictions and observed data assuming Poisson statistics.

        Args:
            outputs (Dict[str, np.ndarray]): The output container to store intermediate results.
            observed_data (Dict[str, np.ndarray]): A dictionary mapping component names to observed data.
            datasets (Dict[str, Dict[str, Union[np.ndarray, float]]]): Input datasets for the model evaluation.
            exposed_variables (Dict[str, Dict[str, Union[np.ndarray, float]]]): Variables exposed by previously evaluated components.
            empty_bins (str): Strategy for handling empty bins (`"skip"` or `"throw"`).

        Returns:
            float: The log-likelihood value.

        Raises:
            ValueError: If empty bins are encountered and `empty_bins="throw"`.
        """
        # Evaluate the analysis
        ana_eval, ana_eval_ssq = self.analysis.evaluate(outputs, datasets, exposed_variables)
        llh = 0.0

        for comp_name, comp_eval in ana_eval.items():
            obs = observed_data.get(comp_name)
            if obs is None:
                raise ValueError(f"No observed data for component '{comp_name}'")

            # Handle empty bins
            empty_expectation = comp_eval == 0
            if empty_bins == "skip":
                comp_eval = comp_eval[~empty_expectation]
                obs = obs[~empty_expectation]
            elif empty_bins == "throw":
                if np.any(empty_expectation):
                    raise ValueError(f"Empty bins in component '{comp_name}'")

            # Compute the log-likelihood
            llh += np.sum(-comp_eval + obs * np.log(comp_eval))

        return llh