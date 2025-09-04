from copy import deepcopy
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import scipy.optimize as sopt
import scipy.stats

from .analysis import Analysis
from .backend import Array, backend
from .likelihood import AbstractLikelihood
from .minimizer import AbstractMinimizer, ScipyMinimizer


class PseudoExpGenerator:
    def __init__(
        self,
        analysis: Analysis,
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        parameter_values: Dict[str, float],
    ):
        """
        Generator for pseudo-experiments based on an analysis.

        Uses the analysis to evaluate expected histograms and then generates
        pseudo-experiments by drawing from a Poisson distribution.

        Args:
            analysis (Analysis): The analysis to generate pseudo-experiments for.
            datasets (Dict[str, Dict[str, Union[Array, float]]]): The input datasets for the model evaluation.
            parameter_values (Dict[str, float]): The parameter values for the analysis.
        """
        self.analysis = analysis
        self.exp_hists, _ = analysis.evaluate(datasets, parameter_values)

    def generate(self, nexp: int) -> Generator[Dict[str, Array], None, None]:
        """
        Generate a pseudo-experiment based on the analysis.

        Returns:
            Dict[str, Array]: A dictionary mapping component names to generated pseudo-experiments.
        """

        for _ in range(nexp):
            obs: Dict[str, Array] = {}
            for key, hist in self.exp_hists.items():
                obs[key] = backend.poiss_rng(hist)
            yield obs


MT = TypeVar("MT", bound=AbstractMinimizer)


class Hypothesis:
    """
    A class to represent a hypothesis for hypothesis testing.

    Default parameter values are taking from the likelihood's seeds.
    Fixed parameters can be provided to create a null hypothesis.
    The minimizer can be customized by providing a different implementation of AbstractMinimizer.
    Args:
        name (str): The name of the hypothesis.
        likelihood (AbstractLikelihood): The likelihood to evaluate the hypothesis.
        fixed_pars (Optional[Dict[str, float]]): Parameters to be fixed in the hypothesis. Defaults to None.
        minimizer (Optional[AbstractMinimizer]): The minimizer to use for fitting. Defaults to ScipyMinimizer.
    """

    def __init__(
        self,
        name: str,
        likelihood: AbstractLikelihood,
        fixed_pars: Optional[Dict[str, float]] = None,
        minimizer: Optional[MT] = None,
    ):
        """
        Initialize the Hypothesis.
        """
        self.name = name
        self.fixed_pars = fixed_pars if fixed_pars is not None else {}
        self.likelihood = likelihood
        self.seeds = likelihood.get_seeds()
        self.minimizer = minimizer or ScipyMinimizer(likelihood)

    @property
    def model_parameters(self) -> Set[str]:
        """
        Get the model parameters for the hypothesis.

        Returns:
            Set[str]: A set of model parameter names.
        """
        return self.likelihood.get_analysis().exposed_parameters - set(
            self.fixed_pars.keys()
        )

    def evaluate(
        self,
        observed_data: Dict[str, Array],
        dataset: Dict[str, Dict[str, Array | float]],
        parameter_values: Optional[Dict[str, float]] = None,
        detailed=False,
    ) -> Union[Array, Tuple[Any, Dict[str, Array], Array]]:
        """
        Evaluate the hypothesis against observed data.

        Args:
            observed_data (Dict[str, Array]): The observed data to evaluate.
            dataset (Dict[str, Dict[str, Array | float]]): The input datasets for the model evaluation.
            parameter_values (Optional[Dict[str, float]]): The parameter values for the hypothesis.
                If not provided, the seeds will be used.
            detailed (bool): If True, return detailed results including the minimizer result.

        Returns:
            float: The log-likelihood value for the hypothesis.
        """

        all_fixed_pars = deepcopy(self.fixed_pars)

        if parameter_values is not None:
            all_fixed_pars.update(parameter_values)
        
        res = self.minimizer.minimize(observed_data, dataset, all_fixed_pars)

        if detailed:
            return res
        else:
            return res[2]  # Return the log-likelihood value

    @property
    def nparams(self) -> int:
        """
        Get the number of parameters in the hypothesis.

        Returns:
            int: The number of parameters.
        """
        return len(self.likelihood.get_analysis().exposed_parameters) - len(
            self.fixed_pars
        )

    def generate_pseudo_experiments(
        self,
        nexp: int,
        dataset: Dict[str, Dict[str, Array | float]],
        parameter_values: Optional[Dict[str, float]] = None,
    ) -> Generator[Dict[str, Array], None, None]:
        """
        Generate pseudo-experiments for the hypothesis.

        Args:
            nexp (int): The number of pseudo-experiments to generate.
            parameter_values (Optional, Dict[str, float]): The parameter values for the hypothesis. If not provided, the seeds will be used.

        Returns:
            Generator[Dict[str, Array], None, None]: A generator yielding pseudo-experiments.
        """

        all_parameter_values = (
            self.seeds | self.fixed_pars if self.fixed_pars else self.seeds
        )
        if parameter_values is not None:
            all_parameter_values.update(parameter_values)

        gen = PseudoExpGenerator(
            self.likelihood.get_analysis(), dataset, all_parameter_values
        )
        return gen.generate(nexp)

    def asimov_experiment(
        self,
        dataset: Dict[str, Dict[str, Array | float]],
        parameter_values: Optional[Dict[str, float]] = None,
    ):
        """
        Generate the Asimov dataset for the hypothesis.

        Args:
            dataset (Dict[str, Dict[str, Array | float]]): The input datasets for the model evaluation.
            parameter_values (Optional[Dict[str, float]]): The parameter values for the hypothesis. If not provided, the seeds will be used.
        Returns:
            Dict[str, Array]: The Asimov dataset.
        """

        all_parameter_values = (
            self.seeds | self.fixed_pars if self.fixed_pars else self.seeds
        )
        if parameter_values is not None:
            all_parameter_values.update(parameter_values)

        hist, hist_ssq = self.likelihood.get_analysis().evaluate(
            dataset, all_parameter_values
        )

        return hist


class HypothesisTest:
    """
    A class to perform hypothesis testing.

    This class encapsulates two hypotheses (null and alternative) and provides methods
    to perform hypothesis tests, generate null and alternative distributions, calculate
    discovery potential, and compute power.

    Args:
        h0 (Hypothesis): The null hypothesis.
        h1 (Hypothesis): The alternative hypothesis.
        dataset (Dict[str, Dict[str, Array | float]]): The input datasets for the model evaluation.
    """

    def __init__(
        self,
        h0: Hypothesis,
        h1: Hypothesis,
        dataset: Dict[str, Dict[str, Array | float]],
    ):
        """
        Initialize the HypothesisTest with two hypotheses.

        Args:
            h0 (Hypothesis): The null hypothesis.
            h1 (Hypothesis): The alternative hypothesis.
        """
        self.h0 = h0
        self.h1 = h1
        self.dataset = dataset

    @classmethod
    def from_likelihood(cls, likelihood: AbstractLikelihood, dataset: Dict[str, Dict[str, Array | float]], fixed_params: Dict[str, float]):
        """
        Create a HypothesisTest from a likelihood.

        Args:
            likelihood (AbstractLikelihood): The likelihood to create hypotheses from.
            dataset (Dict[str, Dict[str, Array | float]]): The input datasets for the model evaluation.
            fixed_params (Dict[str, float]): The parameters to be fixed in the null hypothesis.

        Returns:
            HypothesisTest: The constructed hypothesis test.
        """
        h0 = Hypothesis(
            name="H0",
            likelihood=likelihood,
            fixed_pars=fixed_params,
        )
        h1 = Hypothesis(name="H1", likelihood=likelihood)

        return cls(h0, h1, dataset)


    @property
    def free_parameters(self) -> Set[str]:
        """
        Get the free parameters for the hypothesis test.

        Returns:
            Set[str]: A set of free parameter names.
        """
        return self.h1.model_parameters - self.h0.model_parameters

    @property
    def dof(self) -> int:
        """
        Calculate the degrees of freedom for the hypothesis test.

        Returns:
            int: The degrees of freedom.
        """
        return self.h1.nparams - self.h0.nparams

    def test(
        self,
        observed_data: Dict[str, Array],
        parameter_values: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Perform a hypothesis test against observed data.

        Args:
            observed_data (Dict[str, Array]): The observed data to test against.

        Returns:
            float: The p-value of the hypothesis test.
        """

        llh0 = cast(
            float,
            self.h0.evaluate(
                observed_data, self.dataset, parameter_values, detailed=False
            ),
        )
        llh1 = cast(
            float,
            self.h1.evaluate(
                observed_data, self.dataset, parameter_values, detailed=False
            ),
        )

        dllh = 2 * (llh0 - llh1)

        return dllh

    def null_dist(self, nexp: int) -> List[float]:
        """
        Generate the null distribution for the hypothesis test.

        Args:
            nexp (int): The number of pseudo-experiments to generate.

        Returns:
            List[float]: A list of p-values from the null distribution.
        """

        ts_vals = []

        for exp in self.h0.generate_pseudo_experiments(nexp, self.dataset):
            ts_vals.append(self.test(exp))

        return ts_vals

    def alt_dist(
        self, nexp: int, parameter_values: Optional[Dict[str, float]] = None
    ) -> List[float]:
        """
        Generate the null distribution for the hypothesis test.

        Args:
            nexp (int): The number of pseudo-experiments to generate.

        Returns:
            List[float]: A list of p-values from the null distribution.
        """

        ts_vals = []

        for exp in self.h1.generate_pseudo_experiments(
            nexp, self.dataset, parameter_values
        ):
            ts_vals.append(self.test(exp))

        return ts_vals

    def discovery_potential(
        self,
        nexp_null: int,
        nexp_alt: int,
        sigma_level: float = 3,
        xtol=0.05,
        maxiter=25,
        null_dist: Optional[Array] = None,
    ) -> Tuple[Array, Array, float]:
        """
        Calculate the discovery potential of the hypothesis test.

        Args:
            nexp_null (int): The number of pseudo-experiments for the null hypothesis.
            nexp_alt (int): The number of pseudo-experiments for the alternative hypothesis.
            sigma_level (float, optional): The sigma level for the test. Defaults to 3.
            xtol (float, optional): The tolerance for root finding. Defaults to 0.05.
            maxiter (int, optional): The maximum number of iterations for root finding. Defaults to 25.
            null_dist (Optional[List[float]], optional): Precomputed null distribution. Defaults to None.

        Returns:
            Tuple[Array, Array, float]: A tuple containing the null distribution, alternative distribution, and the parameter value at which the alternative hypothesis is discovered.
        """

        if self.dof != 1:
            raise ValueError(
                "Discovery potential is only implemented for one degree of freedom"
            )

        free_param = list(self.free_parameters)[0]

        # Calc p-value for sigma level
        p_value = 1 - backend.norm_sf(sigma_level)

        if null_dist is None:
            # Generate the null distribution
            null_dist = backend.array(self.null_dist(nexp_null))

        # Calculate the threshold for discovery
        threshold = backend.quantile(null_dist, p_value)

        if threshold == backend.max(null_dist):
            raise ValueError(
                "The null distribution does not contain a value below the threshold. "
                "Increase nexp_null or decrease sigma_level."
            )

        param_bounds = self.h1.likelihood.get_bounds()[free_param]

        def pexp_and_fit(x):
            """
            Generate pseudo-experiments and fit the alternative hypothesis to find the median.
            """
            pexps = self.h1.generate_pseudo_experiments(nexp_alt, self.dataset, {free_param: x})
            alt_dist = backend.array([self.test(exp) for exp in pexps])
            return backend.median(alt_dist)

        alt_param_val = cast(
            float,
            sopt.brentq(
                lambda x: pexp_and_fit(x) - threshold,
                param_bounds[0],
                param_bounds[1],
                xtol=xtol,
                maxiter=maxiter,
            ),
        )

        pexps = self.h1.generate_pseudo_experiments(
            nexp_null, self.dataset, {free_param: alt_param_val}
        )
        alt_dist = backend.array([self.test(exp) for exp in pexps])

        return null_dist, alt_dist, alt_param_val

    def discovery_potential_asimov(
        self, sigma_level: float = 3, xtol=0.05, maxiter=25
    ) -> float:
        """
        Calculate the discovery potential of the hypothesis test using the Asimov dataset.

        Args:
            sigma_level (float, optional): The sigma level for the test. Defaults to 3.

        Returns:
            float: The discovery potential value.
        """

        if self.dof != 1:
            raise ValueError(
                "Discovery potential is only implemented for one degree of freedom"
            )

        free_param = list(self.free_parameters)[0]
        threshold = scipy.stats.chi2.ppf(
            1 - 2 * scipy.stats.norm.sf(sigma_level), self.dof
        )

        def asimov_and_fit(x):
            """
            Generate the Asimov dataset and fit the alternative hypothesis to find the median.
            """
            asimov_exp = self.h1.asimov_experiment(self.dataset, {free_param: x})
            return self.test(asimov_exp)

        param_bounds = self.h1.likelihood.get_bounds()[free_param]

        alt_param_val = cast(
            float,
            sopt.brentq(
                lambda x: asimov_and_fit(x) - threshold,
                param_bounds[0],
                param_bounds[1],
                xtol=xtol,
                maxiter=maxiter,
            ),
        )

        return alt_param_val

    def power(self, nexp: int, sigma_level: float = 3) -> float:
        """
        Calculate the power of the hypothesis test.

        Args:
            nexp (int): The number of pseudo-experiments to generate.
            sigma_level (float, optional): The sigma level for the test. Defaults to 3.

        Returns:
            float: The power of the hypothesis test.
        """

        null_dist = backend.array(self.null_dist(nexp))
        threshold = backend.quantile(null_dist, 1 - backend.norm_sf(sigma_level))

        alt_dist = backend.array(self.alt_dist(nexp))

        # Calculate the power as the fraction of alternative distribution above the threshold
        power = backend.mean(alt_dist > threshold)

        return cast(float, power)

    def scan(
        self,
        observed_data: Dict[str, Array],
        scan_points: int,
    ):

        if self.dof != 1:
            raise ValueError(
                "Scan is only implemented for one degree of freedom"
            )

        free_param = list(self.free_parameters)[0]

        param_bounds = self.h1.likelihood.get_bounds()[free_param]

        scan_grid = np.linspace(param_bounds[0], param_bounds[1], scan_points)
        ts_values = []
        h1_eval = self.h1.evaluate(observed_data, self.dataset, detailed=False)
        for scan_point in scan_grid:
            fp = {free_param: scan_point}
           
            h0_eval = self.h0.evaluate(
                observed_data, self.dataset, fp, detailed=False
            )

            ts = 2 * (cast(float, h0_eval) - cast(float, h1_eval))
            ts_values.append(ts)

        return scan_grid, ts_values


    def uncertainty(self, observed_data, sigma_level):
        if self.dof != 1:
            raise ValueError(
                "Uncertainty calculation is only implemented for one degree of freedom"
        )

        free_param = list(self.free_parameters)[0]
        param_bounds = self.h1.likelihood.get_bounds()[free_param]
        h1_eval = self.h1.evaluate(observed_data, self.dataset, detailed=False)

        delta_llh_thrsh = np.sqrt(sigma_level) # works only for 1 dof

        def fopt(fp):
            h0_eval = self.h0.evaluate(
                observed_data, self.dataset, {free_param: fp}, detailed=False
            )
            ts = 2 * (cast(float, h0_eval) - cast(float, h1_eval))
            return ts - delta_llh_thrsh

        lower = cast(float, sopt.brentq(fopt, param_bounds[0], self.h1.seeds[free_param]))
        upper = cast(float, sopt.brentq(fopt, self.h1.seeds[free_param], param_bounds[1]))

        return (upper - lower) / 2
