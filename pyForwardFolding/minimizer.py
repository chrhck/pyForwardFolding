import functools
from typing import Any, Dict, List, Set, Tuple

import iminuit
from scipy.optimize import Bounds, minimize

from .backend import backend
from .likelihood import AbstractLikelihood, AbstractPrior, GaussianUnivariatePrior


def flat_index_dict_mapping(exp_vars: Set[str], fixed_params: Dict[str, Any] = None) -> Dict[str, Dict[str, int]]:
    """
    Create a mapping of variable names to flat indices for optimization.

    Args:
        exp_vars (Dict[str, List[str]]): Exposed variables.
        fixed_params (Dict[str, Dict[str, Any]], optional): Fixed parameters. Defaults to None.

    Returns:
        Dict[str, Dict[str, int]]: Mapping of variable names to flat indices.
    """
    ix_dict = {}
    for ix, var_name in enumerate(exp_vars):
        if fixed_params and var_name in fixed_params:
            continue
        ix_dict[var_name] = ix

    return ix_dict


def restructure_args(flat_args, exp_vars: Set[str], fixed_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
    for var_name in exp_vars:
        if fixed_params and var_name in fixed_params:
            args[var_name] = fixed_params[var_name]
        else:
            args[var_name] = flat_args[ix]
            ix += 1       
    return args

def destructure_args(params: Dict[str, Any], exp_vars: Set[str], fixed_params: Dict[str, Any] = None) -> List[float]:
    """
    Convert a dictionary of parameters into a flat list.

    Args:
        params (Dict[str, Any]): Parameters to flatten.
        exp_vars (Dict[str, List[str]]): Exposed variables.
        fixed_params (Dict[str, Dict[str, Any]], optional): Fixed parameters. Defaults to None.

    Returns:
        List[float]: Flattened parameters.
    """
    flat_params = []
    for var_name in exp_vars:
        if fixed_params and var_name in fixed_params:
            continue
        flat_params.append(params[var_name])
    return flat_params

class WrappedLLH:
    """
    A wrapper for the likelihood function to flatten parameters for optimization.

    Args:
        likelihood (Callable): The likelihood function to wrap.
        obs (Dict): Observed data.
        datasets (Dict): Input datasets.
        fixed_params (Dict): Parameters to keep fixed during optimization.
    """
    def __init__(self, likelihood: AbstractLikelihood,
                 obs: Dict,
                 datasets: Dict,
                 fixed_params: Dict,
                 prior: List[AbstractPrior]):
        self.likelihood = likelihood
        self.obs = obs
        self.datasets = datasets
        self.fixed_params = fixed_params
        self.prior = prior

    def __call__(self, flat_params) -> float:
        """
        Evaluate the likelihood with flattened parameters.

        Args:
            flat_params: Flattened parameters for optimization.

        Returns:
            float: The negative likelihood value.
        """
        exp_vars = self.likelihood.get_analysis().exposed_parameters
        restructured_args = restructure_args(flat_params,
                                             exp_vars,
                                             self.fixed_params)
        binned_llh = -self.likelihood.llh(self.obs,
                                          self.datasets,
                                          restructured_args)
        for p in self.prior:
            binned_llh += p.log_pdf(restructured_args)
        return binned_llh


class AbstractMinimizer:
    """
    Abstract class for minimizers.

    Args:
        llh (AbstractLikelihood): The likelihood function to minimize.
        obs (Dict): Observed data.
        dataset (Dict): Input datasets.
        exposed_vars: Variables to be optimized.
        bounds: Bounds for the optimization parameters.
        seeds: Initial guesses for the optimization parameters.
        priors (Dict): Prior distributions for the parameters.
        fixed_pars (Dict): Parameters to keep fixed during optimization.
    """
    llh = None
    par_idx_map = None

    def __init__(self,
                 llh: AbstractLikelihood,
                 obs: Dict,
                 dataset: Dict,
                 bounds: Dict[str, Tuple[float, float]],
                 seeds: Dict[str, float],
                 priors: Dict[str, Tuple[float, float]] = {},
                 fixed_pars: Dict[str, float] = {}):
        
        self.llh = llh
        self.obs = obs
        self.dataset = dataset
        self.priors = []
        self.priors.append(GaussianUnivariatePrior(priors))
        self.fixed_pars = fixed_pars

        exposed_vars = self.llh.get_analysis().exposed_parameters

        bounds_list = destructure_args(bounds, exposed_vars, fixed_pars)
        bounds_lower = [bound[0] for bound in bounds_list]
        bounds_upper = [bound[1] for bound in bounds_list]
        seeds_list = destructure_args(seeds, exposed_vars, fixed_pars)
    
        self.bounds = Bounds(bounds_lower, bounds_upper)
        self.seeds = seeds_list

        self.wrapped_lh = WrappedLLH(llh, obs, dataset, fixed_pars, self.priors)


    def minimize(self):
        raise NotImplementedError("Subclasses should implement this method.")


class ScipyMinimizer(AbstractMinimizer):
    def __init__(self,
                 llh: AbstractLikelihood,
                 obs: dict,
                 dataset: dict,
                 bounds: Dict[str, Dict[str, Tuple[float, float]]],
                 seeds: Dict[str, Dict[str, float]],
                 priors: Dict[str, Dict[str, Tuple[float, float]]] = {},
                 fixed_pars: Dict[str, Dict[str, float]] = {},
                 tol: float = 1E-10):
        
        super().__init__(
            llh=llh,
            obs=obs,
            dataset=dataset,
            bounds=bounds,
            seeds=seeds,
            priors=priors,
            fixed_pars=fixed_pars)
        
        self.fmin_and_grad = backend.func_and_grad(self.wrapped_lh)
        self.tol = tol

    def minimize(self):
        result = minimize(
            self.fmin_and_grad,
            self.seeds,
            bounds=self.bounds,
            jac=True,
            method="L-BFGS-B",
            tol=self.tol,
            options={"maxls": 50, "maxcor": 50}
            )
        
        res_dict = restructure_args(
            result.x,
            self.llh.get_analysis().exposed_parameters,
            self.fixed_pars
        )

        return result, res_dict, result.fun


def wrap_func_np_cache(func):

    def wrapper(np_array):
        hashable_array = tuple(np_array)
        return cached_wrap(hashable_array)

    @functools.lru_cache(maxsize=None)
    def cached_wrap(hashable_arr):
        arr = backend.array(hashable_arr)
        return func(arr)
    
    return wrapper
    



class MinuitMinimizer(AbstractMinimizer):
    def __init__(self,
                 llh: AbstractLikelihood,
                 obs: dict,
                 dataset: dict,
                 bounds: Dict[str, Dict[str, Tuple[float, float]]],
                 seeds: Dict[str, Dict[str, float]],
                 priors: Dict[str, Dict[str, Tuple[float, float]]] = {},
                 fixed_pars: Dict[str, Dict[str, float]] = {},
                 simplex_prefit: bool = False):

        super().__init__(
            llh=llh,
            obs=obs,
            dataset=dataset,
            bounds=bounds,
            seeds=seeds,
            priors=priors,
            fixed_pars=fixed_pars)

        self.func = wrap_func_np_cache(backend.compile(self.wrapped_lh))
        self.grad = wrap_func_np_cache(backend.compile(backend.grad(self.wrapped_lh)))

        names = [par_name for par_name in self.llh.get_analysis().exposed_parameters if par_name not in self.fixed_pars]

        self.minuit = iminuit.Minuit(
            self.func,
            self.seeds,
            grad=self.grad,
            name=names
        )

        bound_list = [[lb, ub] for lb, ub in zip(self.bounds.lb, self.bounds.ub)]
        self.minuit.errordef = self.minuit.LIKELIHOOD
        self.minuit.limits = bound_list
        self.minuit.strategy = 1
        self.minuit.tol = 1e-2
        self.minuit.print_level = 0

        self.simplex_prefit = simplex_prefit

    def _build_message(self):
        """
        Helper function for building a short fit message.
        """
        if self.minuit.valid:
            message = "Optimization terminated successfully"
            if self.minuit.accurate:
                message += "."
            else:
                message += ", but uncertainties are unrealiable."
        else:
            message = "Optimization failed."
            fmin = self.minuit.fmin
            if fmin is not None:
                if fmin.has_reached_call_limit:
                    message += " Call limit was reached."
                if fmin.is_above_max_edm:
                    message += " Estimated distance to minimum too large."

        return message

    def minimize(self):
        
        if self.simplex_prefit:
            self.minuit.simplex().migrad()
        else:
            self.minuit.migrad()
            
        minimizer_info = {
            'success': self.minuit.valid,
            'message': self._build_message(),
            'nfev': self.minuit.nfcn,
            'njev': self.minuit.ngrad,
            'hess_inv': self.minuit.covariance,
        }

        res_dict = restructure_args(
            self.minuit.values,
            self.llh.get_analysis().exposed_parameters,
            self.fixed_pars
        )

       
        fun = self.minuit.fval
        print("best-fit llh: ", fun)
        print("----------------")
        print("best-fit parameters:")
        for key, val in res_dict.items():
            print(f"{key}: {val}")
        print("----------------")

        return minimizer_info, res_dict, fun
