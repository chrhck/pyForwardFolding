from typing import Callable, Dict, Any, Tuple, List
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize


def flat_index_dict_mapping(exp_vars: Dict[str, List[str]], fixed_params: Dict[str, Dict[str, Any]] = None) -> Dict[str, Dict[str, int]]:
    """
    Create a mapping of variable names to flat indices for optimization.

    Args:
        exp_vars (Dict[str, List[str]]): Exposed variables.
        fixed_params (Dict[str, Dict[str, Any]], optional): Fixed parameters. Defaults to None.

    Returns:
        Dict[str, Dict[str, int]]: Mapping of variable names to flat indices.
    """
    ix_dict = {}
    ix = 0
    for var_name, vars_exp in exp_vars.items():
        if var_name not in ix_dict:
            ix_dict[var_name] = {}
        for var_exp in vars_exp:
            if fixed_params and var_name in fixed_params and var_exp in fixed_params[var_name]:
                continue
            ix_dict[var_name][var_exp] = ix
            ix += 1
    return ix_dict


def restructure_args(flat_args, exp_vars: Dict[str, List[str]], fixed_params: Dict[str, Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Restructure flat arguments into the original parameter structure.

    Args:
        flat_args: Flattened arguments.
        exp_vars (Dict[str, List[str]]): Exposed variables.
        fixed_params (Dict[str, Dict[str, Any]], optional): Fixed parameters. Defaults to None.

    Returns:
        Dict[str, Dict[str, Any]]: Restructured arguments.
    """
    args = {}
    ix = 0
    for var_name, vars_exp in exp_vars.items():
        args[var_name] = {}
        for var_exp in vars_exp:
            if fixed_params and var_name in fixed_params and var_exp in fixed_params[var_name]:
                args[var_name][var_exp] = fixed_params[var_name][var_exp]
            else:
                args[var_name][var_exp] = flat_args[ix]
                ix += 1
    return args


class WrappedLLH:
    """
    A wrapper for the likelihood function to flatten parameters for optimization.

    Args:
        likelihood (Callable): The likelihood function to wrap.
        obs (Dict): Observed data.
        datasets (Dict): Input datasets.
        fixed_params (Dict): Parameters to keep fixed during optimization.
    """
    def __init__(self, likelihood: Callable, obs: Dict, datasets: Dict, fixed_params: Dict):
        self.likelihood = likelihood
        self.obs = obs
        self.datasets = datasets
        self.fixed_params = fixed_params

    def __call__(self, flat_params) -> float:
        """
        Evaluate the likelihood with flattened parameters.

        Args:
            flat_params: Flattened parameters for optimization.

        Returns:
            float: The negative likelihood value.
        """
        exp_vars = self.likelihood.get_analysis().exposed_variables()
        restructured_args = restructure_args(flat_params, exp_vars, self.fixed_params)
        return -self.likelihood.llh(self.obs, self.datasets, restructured_args)