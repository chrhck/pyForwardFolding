from typing import Dict, Set, Tuple, Union

import numpy as np
from jax import jacfwd, tree_util
import jax.numpy as jnp

from .binned_expectation import BinnedExpectation


class Analysis:
    """
    Represents a complete analysis consisting of multiple binned expectations.

    Args:
        expectations (Dict[str, BinnedExpectation]): Dictionary mapping expectation names to their objects.
    """
    def __init__(self, expectations: Dict[str, BinnedExpectation]):
        self.expectations = expectations

    @property
    def required_variables(self) -> Set[str]:
        """
        Get all variables required by any component in the analysis.

        Returns:
            Set[str]: A set containing all required variables.
        """
        return set.union(*(comp.required_variables for comp in self.expectations.values()))

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get all parameters exposed by the analysis expectations.

        Returns:
            Set[str]: A set of all exposed parameters from all expectations.
        """
        exposed = set()
        for comp in self.expectations.values():
            exposed |= comp.exposed_parameters
        return exposed

    def evaluate(
        self,
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        parameter_values: Dict[str, Union[np.ndarray, float]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Evaluate all expectations in the analysis.

        Args:
            datasets (Dict[str, Dict[str, Union[np.ndarray, float]]]): A dictionary mapping component names to their input variables.
            parameter_values (Dict[str, Union[np.ndarray, float]]): Variables exposed by previously evaluated expectations.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: A tuple containing:
                - A dictionary mapping component names to their evaluation results (histograms).
                - A dictionary mapping component names to their squared evaluation results (histograms).
        """
        output_dict = {}
        output_ssq_dict = {}

        for comp_name, comp in self.expectations.items():
           
            # Evaluate the component
            hist, hist_ssq = comp.evaluate(
                datasets,
                parameter_values,
            )

            # Store results
            output_dict[comp_name] = hist
            output_ssq_dict[comp_name] = hist_ssq

        return output_dict, output_ssq_dict
    
    def fisher_information(
        self,
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        parameter_values: Dict[str, Union[np.ndarray, float]],
    ) -> jnp.ndarray:
        """
        Calculate the Fisher Information matrix at the given parameter values.

        The Fisher Information quantifies the amount of information that the observed data
        carries about the model parameters. It is computed by evaluating the gradient of
        the expected bin counts with respect to the parameters and applying the standard
        Fisher Information formula when assuming a Poisson likelihood:
        
            I_ij = Σ_k (∂μ_k/∂θ_i ∂μ_k/∂θ_j) / μ_k

        where μ_k is the expected count in bin k, and θ_i, θ_j are parameters.

        Args:
            datasets (Dict[str, Dict[str, Union[np.ndarray, float]]]): A dictionary mapping 
                component names to their input variables.
            parameter_values (Dict[str, Union[np.ndarray, float]]): A dictionary mapping 
                parameter names to their values.

        Returns:
            jnp.ndarray: The Fisher Information matrix (shape: [n_params, n_params]).
        """
        fisher_dict = {}
        for comp_name, comp in self.expectations.items():
            # Bad to evaluate comp.evaluate twice, but unfortunately there is no jacfwd_and_value...
            grads, _ = jacfwd(comp.evaluate, argnums=1, has_aux=True)(
                datasets,
                parameter_values,
            )
            hist, _ = comp.evaluate(
                datasets,
                parameter_values,
            )

            # As jacfwd does destroy the key ordering, we need to loop over the keys of parameters values
            # This will later help to keep track of which variance belongs to which parameter
            flat_grads = [grads[k].flatten() for k in parameter_values]
            hist = hist.flatten()
            information = [jnp.where(hist == 0, 0.0, g / jnp.sqrt(hist)) for g in flat_grads]
            values = jnp.stack(information)
            fisher_information = values @ values.T
            fisher_dict[comp_name] = fisher_information

        return jnp.sum(jnp.asarray([v for v in fisher_dict.values()]),axis=0)

    def covariance(
        self,
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        parameter_values: Dict[str, Union[np.ndarray, float]],
    ) -> jnp.ndarray:
        """
        Compute the covariance matrix of the parameters by inverting the Fisher Information matrix.

        This matrix represents the best-case lower bound on the covariance of any unbiased
        estimator of the parameters (the Cramér–Rao bound).

        Args:
            datasets (Dict[str, Dict[str, Union[np.ndarray, float]]]): A dictionary mapping 
                component names to their input variables.
            parameter_values (Dict[str, Union[np.ndarray, float]]): A dictionary mapping 
                parameter names to their values.

        Returns:
            jnp.ndarray: The parameter covariance matrix (shape: [n_params, n_params]).
        """
        fisher_information = self.fisher_information(datasets, parameter_values)
        cov = jnp.linalg.inv(fisher_information)
        return cov

    def variance(
        self,
        datasets: Dict[str, Dict[str, Union[np.ndarray, float]]],
        parameter_values: Dict[str, Union[np.ndarray, float]],
    ) -> Dict[str, jnp.ndarray]:
        """
        Extract the variances (diagonal elements) from the parameter covariance matrix.

        Each variance corresponds to the square of the minimal achievable standard deviation
        of an unbiased estimator for that parameter.

        Args:
            datasets (Dict[str, Dict[str, Union[np.ndarray, float]]]): A dictionary mapping 
                component names to their input variables.
            parameter_values (Dict[str, Union[np.ndarray, float]]): A dictionary mapping 
                parameter names to their values.

        Returns:
            Dict[str, jnp.ndarray]: A dictionary mapping parameter names to their variances.
        """
        cov = self.covariance(datasets, parameter_values)
        var = jnp.diag(cov)
        keys = parameter_values.keys()

        return {
            k: v for k, v in zip(keys, var)
        }

    def __repr__(self):
        """
        String representation of the Analysis object.

        Returns:
            str: A string representation of the analysis.
        """
        return f"Analysis with {len(self.expectations)} expectations: {', '.join(self.expectations.keys())}"
    
    def __getitem__(self, item: str) -> BinnedExpectation:
        """
        Get a specific expectation by name.

        Args:
            item (str): The name of the expectation.

        Returns:
            BinnedExpectation: The corresponding expectation object.
        """
        return self.expectations[item]
    
