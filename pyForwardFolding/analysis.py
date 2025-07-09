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
        Fisher Information formula:
        
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
        for comp_name, comp in self.expectations.items():
            grads, _ = jacfwd(comp.evaluate, argnums=1, has_aux=True)(
                datasets,
                parameter_values,
            )
            hist, _ = comp.evaluate(
                datasets,
                parameter_values,
            )
            grads = tree_util.tree_map(lambda v: v.flatten(), grads)
            grads = {k: grads[k] for k in parameter_values.keys()} # Reorder keys, as jacfwd does NOT keep key order
            hist = hist.flatten()

            information = tree_util.tree_map(lambda g: jnp.where(hist == 0, 0.0, g / jnp.sqrt(hist)), grads)
            flat_values, _ = tree_util.tree_flatten(information)
            values = jnp.stack(flat_values)

            fisher_information = values[:, None, :] * values[None, :, :]
            fisher_information = jnp.sum(fisher_information, axis=-1)

        return fisher_information

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