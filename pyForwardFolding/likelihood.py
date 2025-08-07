from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

from .analysis import Analysis
from .backend import Array, backend


class AbstractPrior:
    def __init__(
        self,
        prior_seeds: Dict[str, float],
        prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Abstract base class for priors used in likelihood evaluations.

        Args:
            prior_seeds (Dict[str, float]): Initial values for the parameters.
            prior_bounds (Optional[Dict[str, Tuple[float, float]]]): Optional bounds for the parameters.
        """
        self.prior_seeds = prior_seeds

        if prior_bounds is None:
            prior_bounds = {par: (-np.inf, np.inf) for par in prior_seeds.keys()}
        self.prior_bounds = prior_bounds

    def log_pdf(self, parameter_values: Dict[str, float]) -> float:
        raise NotImplementedError

    @property
    def prior_variables(self) -> Set[str]:
        """
        Get the set of prior variables.

        Returns:
            Set[str]: A set of all prior variable names.
        """
        return set(self.prior_seeds.keys())


class GaussianUnivariatePrior(AbstractPrior):
    def __init__(
        self,
        prior_params: Dict[str, Tuple[float, float]],
        prior_seeds: Dict[str, float],
        prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        A univariate Gaussian prior for parameters.

        Note, this prior is not normalized, it only provides a log likelihood contribution.

        Args:
            prior_params (Dict[str, Tuple[float, float]]): Dictionary mapping parameter names to (mean, std) tuples.
            prior_seeds (Dict[str, float]): Initial values for the parameters.
            prior_bounds (Optional[Dict[str, Tuple[float, float]]]): Optional bounds for the parameters.
        """
        super().__init__(prior_seeds, prior_bounds)
        self.prior_params = prior_params

    def log_pdf(self, parameter_values):
        llh = 0
        for par, (mean, std) in self.prior_params.items():
            llh += -((parameter_values[par] - mean) ** 2) / (2 * std**2)
        return llh


class UniformPrior(AbstractPrior):
    def __init__(
        self,
        prior_seeds: Dict[str, float],
        prior_bounds: Dict[str, Tuple[float, float]],
    ):
        """
        A uniform prior for parameters.

        Note, this is essentially a dummy prior. No bounds checkinf is performed.

        Args:
            prior_seeds (Dict[str, float]): Initial values for the parameters.
            prior_bounds (Optional[Dict[str, Tuple[float, float]]]): Bounds for the parameters.
        """
        super().__init__(prior_seeds, prior_bounds)

    def log_pdf(self, parameter_values):
        return 0.0


T = TypeVar("T", bound=AbstractPrior)


class AbstractLikelihood:
    """
    Abstract base class representing a likelihood function to be evaluated against observed data.
    """

    def __init__(self, analysis: Analysis, priors: List[T]):
        # Check that priors are set for all exposed parameters
        exposed = analysis.exposed_parameters

        prior_variables = set()
        for prior in priors:
            prior_variables.update(prior.prior_variables)

        if exposed != prior_variables:
            raise ValueError("Mismatch between exposed parameters and prior variables.")

        self.analysis = analysis
        self.priors = priors

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
        parameter_values: Dict[str, float],
        empty_bins: str = "skip",
    ) -> Array:
        raise NotImplementedError

    def get_seeds(self) -> Dict[str, float]:
        """
        Get the seeds for the parameters defined by the priors.

        Returns:
            Dict[str, float]: A dictionary mapping parameter names to their initial values.
        """
        seeds = {}
        for prior in self.priors:
            seeds.update(prior.prior_seeds)
        return seeds
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters defined by the priors.

        Returns:
            Dict[str, Tuple[float, float]]: A dictionary mapping parameter names to their bounds.
        """
        bounds = {}
        for prior in self.priors:
            bounds.update(prior.prior_bounds)
        return bounds


class PoissonLikelihood(AbstractLikelihood):
    """
    A likelihood function that assumes Poisson-distributed data.

    Args:
        analysis (Analysis): The analysis to evaluate against observed data.
    """

    def __init__(self, analysis: Analysis, priors: List[T]):
        super().__init__(analysis, priors)

    def llh(
        self,
        observed_data: Dict[str, Array],
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        parameter_values: Dict[str, float],
        empty_bins: str = "skip",
    ) -> Array:
        """
        Compute the log-likelihood between model predictions and observed data assuming Poisson statistics.

        Args:
            observed_data (Dict[str, Array]): A dictionary mapping component names to observed data.
            datasets (Dict[str, Dict[str, Union[Array, float]]]): Input datasets for the model evaluation.
            parameter_values (Dict[str, float]): Variables exposed by previously evaluated components.
            empty_bins (str): Strategy for handling empty bins (`"skip"` or `"throw"`).

        Returns:
            Array: The log-likelihood value.

        Raises:
            ValueError: If empty bins are encountered and `empty_bins="throw"`.
        """
        # Evaluate the analysis
        ana_eval, ana_eval_ssq = self.analysis.evaluate(datasets, parameter_values)
        llh = backend.array(0.0)

        for comp_name, comp_eval in ana_eval.items():
            obs = observed_data.get(comp_name)
            if obs is None:
                raise ValueError(f"No observed data for component '{comp_name}'")
            # Handle empty bins
            non_empty_expectation = comp_eval > 0
            # non_empty_observations = obs > 0
            if empty_bins == "skip":
                comp_eval_shift = backend.select(non_empty_expectation, comp_eval, 1e-8)
                # obs_shift = backend.select(non_empty_observations,
                #                           obs,
                #                           obs + 1e-8)
                llh_bins = backend.where_sum(
                    non_empty_expectation,
                    -comp_eval_shift
                    + obs * backend.log(comp_eval_shift)
                    - backend.gammaln(obs + 1),
                    0,
                )
                # llh_sat = backend.where_sum(non_empty_observations,
                #                            -obs + obs * backend.log(obs_shift),
                #                            0)
                # llh += (llh_bins - llh_sat)
                llh += llh_bins
            elif empty_bins == "throw":
                if not np.all(non_empty_expectation):
                    raise ValueError(f"Empty bins in component '{comp_name}'")

        for p in self.priors:
            llh += p.log_pdf(parameter_values)
        return llh


class SAYLikelihood(AbstractLikelihood):
    """
    Extension of the Poisson likelihood to account for limited MC statistics.

    https://doi.org/10.48550/arXiv.1901.04645
    """

    def __init__(self, analysis: Analysis, priors: List[T]):
        super().__init__(analysis, priors)

    def llh(
        self,
        observed_data: Dict[str, Array],
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        parameter_values: Dict[str, float],
        empty_bins: str = "skip",
    ) -> Array:
        # Evaluate the analysis
        ana_eval, ana_eval_ssq = self.analysis.evaluate(datasets, parameter_values)
        total_llh = backend.array(0.0)

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

            sanitized_ssq = backend.select(comp_ssq > 0, comp_ssq, 1e-8)
            sanitized = backend.select(non_empty_expectation, comp_eval, 1e-8)

            alpha = sanitized**2 / sanitized_ssq + 1.0
            beta = sanitized / sanitized_ssq

            llh_eff = (
                alpha * backend.log(beta)
                + backend.gammaln(obs + alpha)
                - backend.gammaln(obs + 1)
                - (obs + alpha) * backend.log(1.0 + beta)
                - backend.gammaln(alpha)
            )

            llh_poisson = (
                -sanitized + obs * backend.log(sanitized) - backend.gammaln(obs + 1)
            )

            llh = backend.where(sanitized_ssq > 0, llh_eff, llh_poisson)

            llh_sum = backend.where_sum(non_empty_expectation, llh, 0)

            total_llh += llh_sum

        for p in self.priors:
            total_llh += p.log_pdf(parameter_values)
        return total_llh
