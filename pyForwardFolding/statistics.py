from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union, cast

import scipy.optimize as sopt

from .analysis import Analysis
from .backend import Array, backend
from .likelihood import AbstractLikelihood
from .minimizer import ScipyMinimizer


class PseudoExpGenerator:
    def __init__(
        self,
        analysis: Analysis,
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        parameter_values: Dict[str, float],
    ):
        """
        Initialize the PseudoExpGenerator with an analysis.

        Args:
            analysis (Analysis): The analysis to generate pseudo-experiments for.
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


class Hypothesis:
    """
    A class to represent a hypothesis for hypothesis testing.
    """

    def __init__(
        self,
        name: str,
        likelihood: AbstractLikelihood,
        datasets: Dict,
        bounds: Dict[str, Tuple[float, float]],
        seeds: Dict[str, float],
        priors: Optional[Dict[str, Tuple[float, float]]] = None,
        fixed_pars: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the Hypothesis.
        """
        self.name = name
        self.datasets = datasets
        self.fixed_pars = fixed_pars
        self.likelihood = likelihood
        self.priors = priors
        self.bounds = bounds
        self.seeds = seeds

    @property
    def model_parameters(self) -> Set[str]:
        """
        Get the model parameters for the hypothesis.

        Returns:
            Set[str]: A set of model parameter names.
        """
        return (
            self.likelihood.get_analysis().exposed_parameters
            - set(self.fixed_pars.keys())
            if self.fixed_pars
            else self.likelihood.get_analysis().exposed_parameters
        )

    def evaluate(
        self, observed_data: Dict[str, Array], detailed=False
    ) -> Union[Array, Tuple[Any, Dict[str, Array], Array]]:
        """
        Evaluate the hypothesis against observed data.

        Args:
            observed_data (Dict[str, Array]): The observed data to evaluate.

        Returns:
            float: The log-likelihood value for the hypothesis.
        """

        mini = ScipyMinimizer(
            self.likelihood,
            observed_data,
            self.datasets,
            self.bounds,
            self.seeds,
            self.priors,
            self.fixed_pars,
        )

        if detailed:
            return mini.minimize()
        else:
            return mini.minimize()[2]  # Return the log-likelihood value

    @property
    def nparams(self) -> int:
        """
        Get the number of parameters in the hypothesis.

        Returns:
            int: The number of parameters.
        """
        return (
            len(self.likelihood.get_analysis().exposed_parameters)
            - len(self.fixed_pars)
            if self.fixed_pars
            else len(self.likelihood.get_analysis().exposed_parameters)
        )

    def generate_pseudo_experiments(
        self, nexp: int, parameter_values: Optional[Dict[str, float]] = None
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
            self.likelihood.get_analysis(), self.datasets, all_parameter_values
        )
        return gen.generate(nexp)

    def asimov_experiment(self):
        parameter_values = (
            self.seeds | self.fixed_pars if self.fixed_pars else self.seeds
        )

        hist, hist_ssq = self.likelihood.get_analysis().evaluate(
            self.datasets, parameter_values
        )

        return hist


class HypothesisTest:
    """
    A class to perform hypothesis testing using pseudo-experiments.
    """

    def __init__(self, h0: Hypothesis, h1: Hypothesis):
        """
        Initialize the HypothesisTest with two hypotheses.

        Args:
            h0 (Hypothesis): The null hypothesis.
            h1 (Hypothesis): The alternative hypothesis.
        """
        self.h0 = h0
        self.h1 = h1

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

    def test(self, observed_data: Dict[str, Array]) -> float:
        """
        Perform a hypothesis test against observed data.

        Args:
            observed_data (Dict[str, Array]): The observed data to test against.

        Returns:
            float: The p-value of the hypothesis test.
        """

        llh0 = cast(float, self.h0.evaluate(observed_data, detailed=False))
        llh1 = cast(float, self.h1.evaluate(observed_data, detailed=False))

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

        for exp in self.h0.generate_pseudo_experiments(nexp):
            ts_vals.append(self.test(exp))

        return ts_vals

    def alt_dist(self, nexp: int) -> List[float]:
        """
        Generate the null distribution for the hypothesis test.

        Args:
            nexp (int): The number of pseudo-experiments to generate.

        Returns:
            List[float]: A list of p-values from the null distribution.
        """

        ts_vals = []

        for exp in self.h1.generate_pseudo_experiments(nexp):
            ts_vals.append(self.test(exp))

        return ts_vals

    def discovery_potential(
        self,
        nexp_null: int,
        nexp_alt: int,
        sigma_level: float = 3,
        xtol=0.05,
        maxiter=25,
    ) -> Tuple[Array, Array, float]:
        if self.dof != 1:
            raise ValueError(
                "Discovery potential is only implemented for one degree of freedom"
            )

        free_param = list(self.free_parameters)[0]

        # Calc p-value for sigma level
        p_value = 1 - backend.norm_sf(sigma_level)

        null_dist = backend.array(self.null_dist(nexp_null))

        # Calculate the threshold for discovery
        threshold = backend.quantile(null_dist, p_value)

        param_bounds = self.h1.bounds[free_param]

        def pexp_and_fit(x):
            """
            Generate pseudo-experiments and fit the alternative hypothesis to find the median.
            """
            pexps = self.h1.generate_pseudo_experiments(nexp_alt, {free_param: x})
            alt_dist = backend.array([self.test(exp) for exp in pexps])
            return backend.median(alt_dist)

        alt_param_val = cast(float, sopt.brentq(
            lambda x: pexp_and_fit(x) - threshold,
            param_bounds[0],
            param_bounds[1],
            xtol=xtol,
            maxiter=maxiter,
        ))

        pexps = self.h1.generate_pseudo_experiments(
            nexp_null, {free_param: alt_param_val}
        )
        alt_dist = backend.array([self.test(exp) for exp in pexps])

        return null_dist, alt_dist, alt_param_val
