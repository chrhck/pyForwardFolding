from typing import Dict, Tuple, Union

import numpy as np

from .analysis import Analysis
from .backend import Array, backend


class AbstractLikelihood:
    """
    Abstract base class representing a likelihood function to be evaluated against observed data.
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
        observed_data: Dict[str, Array],
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        exposed_variables: Dict[str, Dict[str, Union[Array, float]]],
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
        super().__init__(analysis)
       
    def llh(
        self,
        observed_data: Dict[str, Array],
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        exposed_variables: Dict[str, Dict[str, Union[Array, float]]],
        empty_bins: str = "skip",
    ) -> float:
        """
        Compute the log-likelihood between model predictions and observed data assuming Poisson statistics.

        Args:
            observed_data (Dict[str, Array]): A dictionary mapping component names to observed data.
            datasets (Dict[str, Dict[str, Union[Array, float]]]): Input datasets for the model evaluation.
            exposed_variables (Dict[str, Dict[str, Union[Array, float]]]): Variables exposed by previously evaluated components.
            empty_bins (str): Strategy for handling empty bins (`"skip"` or `"throw"`).

        Returns:
            float: The log-likelihood value.

        Raises:
            ValueError: If empty bins are encountered and `empty_bins="throw"`.
        """
        # Evaluate the analysis
        ana_eval, ana_eval_ssq = self.analysis.evaluate(datasets, exposed_variables)
        llh = 0.0

        for comp_name, comp_eval in ana_eval.items():
            obs = observed_data.get(comp_name)
            if obs is None:
                raise ValueError(f"No observed data for component '{comp_name}'")
            # Handle empty bins
            non_empty_expectation = comp_eval > 0
            #non_empty_observations = obs > 0
            if empty_bins == "skip":
                comp_eval_shift = backend.select(non_empty_expectation,
                                                 comp_eval,
                                                 1E-8)
                #obs_shift = backend.select(non_empty_observations,
                #                           obs,
                #                           obs + 1e-8)
                llh_bins = backend.where_sum(non_empty_expectation,
                                             -comp_eval_shift + obs * backend.log(comp_eval_shift) - backend.gammaln(obs + 1),
                                             0)
                #llh_sat = backend.where_sum(non_empty_observations,
                #                            -obs + obs * backend.log(obs_shift),
                #                            0)
                #llh += (llh_bins - llh_sat)
                llh += llh_bins
            elif empty_bins == "throw":
                if not np.all(non_empty_expectation):
                    raise ValueError(f"Empty bins in component '{comp_name}'")
        return llh


class AbstractPrior:
    def log_pdf(self, exposed_variables):
        raise NotImplementedError


class GaussianUnivariatePrior(AbstractPrior):
    def __init__(self, prior_params: Dict[str, Tuple[float, float]]):
        self.prior_params = prior_params


    def log_pdf(self, exposed_parameters):
        llh = 0
        for par, (mean, std) in self.prior_params.items():
            llh += (exposed_parameters[par] - mean)**2 / std**2
        return llh


class SAYLikelihood(AbstractLikelihood):
    """
    Extension of the Poisson likelihood to account for limited MC statistics.

    https://doi.org/10.48550/arXiv.1901.04645    
    """

    def __init__(self, analysis: Analysis):
        self.analysis = analysis

    def llh(
        self,
        observed_data: Dict[str, Array],
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        exposed_variables: Dict[str, Dict[str, Union[Array, float]]],
        empty_bins: str = "skip",
    ) -> float:
        # Evaluate the analysis
        ana_eval, ana_eval_ssq = self.analysis.evaluate(datasets, exposed_variables)
        total_llh = 0.0

        for comp_name, comp_eval in ana_eval.items():
            obs = observed_data.get(comp_name)
            if obs is None:
                raise ValueError(f"No observed data for component '{comp_name}'")
            
            
            non_empty_expectation = comp_eval > 0
            # Handle empty bins
            if empty_bins == "throw":
                if not np.all(non_empty_expectation):
                    raise ValueError(f"Empty bins in component '{comp_name}'")

            comp_ssq = ana_eval_ssq[comp_name]
            # Clip SSQ (which could be >mu^2 due to nuisance parameters)
            comp_ssq = backend.clip(comp_ssq, 0, comp_eval**2)

            sanitized_ssq = backend.select(comp_ssq > 0, comp_ssq, 1E-8)
            sanizized_mu = backend.select(non_empty_expectation,
                                          comp_eval,
                                          1E-8)

            alpha = sanizized_mu**2 / sanitized_ssq + 1.0
            beta = sanizized_mu / sanitized_ssq

            llh_eff = alpha * backend.log(beta) + backend.gammaln(obs+alpha) - \
                backend.gammaln(obs + 1) - (obs+alpha)*backend.log(1.+beta) - \
                backend.gammaln(alpha)
            
            llh_poisson =  -sanizized_mu + obs * backend.log(sanizized_mu) - backend.gammaln(obs + 1)

            llh = backend.where(
                 sanitized_ssq > 0,
                 llh_eff,
                 llh_poisson
            )

            llh_sum = backend.where_sum(
                non_empty_expectation,
                llh,
               0
            )

            total_llh += llh_sum
        return total_llh 

     