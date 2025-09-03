import functools
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, cast

import iminuit
from scipy.optimize import Bounds, minimize

from .backend import Array, backend
from .likelihood import AbstractLikelihood


def flat_index_dict_mapping(
    exp_vars: Set[str], fixed_params: Optional[Dict[str, Any]] = None
) -> Dict[str, int]:
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


def restructure_args(
    flat_args, exp_vars: List[str], fixed_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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


T = TypeVar("T")


def destructure_args(
    params: Dict[str, T],
    exp_vars: List[str],
    fixed_params: Optional[Dict[str, Any]] = None,
) -> List[T]:
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

    def __init__(
        self,
        likelihood: AbstractLikelihood,
        obs: Dict,
        datasets: Dict,
        fixed_params: Dict,
    ):
        self.likelihood = likelihood
        self.obs = obs
        self.datasets = datasets
        self.fixed_params = fixed_params
        self.parameters = sorted(
            list(self.likelihood.get_analysis().exposed_parameters)
        )

    def __call__(self, flat_params) -> Array:
        """
        Evaluate the likelihood with flattened parameters.

        Args:
            flat_params: Flattened parameters for optimization.

        Returns:
            float: The negative likelihood value.
        """
        restructured_args = restructure_args(
            flat_params, self.parameters, self.fixed_params
        )
        binned_llh = self.likelihood.llh(self.obs, self.datasets, restructured_args)
        return -binned_llh


class AbstractMinimizer:
    """
    Abstract class for minimizers.

    Args:
        llh (AbstractLikelihood): The likelihood function to minimize.
        obs (Dict): Observed data.
        dataset (Dict): Input datasets.
        fixed_pars (Dict): Parameters to keep fixed during optimization.
    """

    def __init__(
        self,
        llh: AbstractLikelihood,
    ):
        self.llh = llh
        self.exposed_vars = self.llh.get_analysis().exposed_parameters

        self.parameters = sorted(list(self.exposed_vars))

    def make_bounds(self, fixed_pars: Optional[Dict[str, float]] = None) -> Bounds:
        """
        Create bounds for the optimization.
        """
        bounds_dict = {}
        for prior in self.llh.priors:
            bounds_dict.update(prior.prior_bounds)

        bounds_list = destructure_args(bounds_dict, self.parameters, fixed_pars)
        bounds_lower = [bound[0] for bound in bounds_list]
        bounds_upper = [bound[1] for bound in bounds_list]
        return Bounds(bounds_lower, bounds_upper)  # type: ignore

    def make_seeds(self, fixed_pars: Optional[Dict[str, float]] = None) -> List[float]:
        """
        Create seeds for the optimization.
        """
        seeds_dict = {}
        for prior in self.llh.priors:
            seeds_dict.update(prior.prior_seeds)

        return destructure_args(seeds_dict, self.parameters, fixed_pars)

    def minimize(
        self,
        obs: Dict,
        dataset: Dict,
        fixed_pars: Optional[Dict[str, float]] = None,
    ) -> Tuple[Any, Dict[str, Any], Array]:
        raise NotImplementedError("Subclasses should implement this method.")


class ScipyMinimizer(AbstractMinimizer):
    def __init__(
        self,
        llh: AbstractLikelihood,
        tol: float = 1e-10,
    ):
        super().__init__(
            llh=llh,
        )

        self.tol = tol

    def minimize(
        self,
        obs: Dict,
        dataset: Dict,
        fixed_pars: Optional[Dict[str, float]] = None,
    ) -> Tuple[Any, Dict[str, Any], Array]:
        if fixed_pars is None:
            fixed_pars = {}

        wrapped_lh = WrappedLLH(self.llh, obs, dataset, fixed_pars)
        fmin_and_grad = backend.func_and_grad(wrapped_lh)

        all_fixed = set(fixed_pars.keys()) == self.exposed_vars

        if all_fixed:
            flat_params = destructure_args(fixed_pars, self.parameters)
            result = wrapped_lh(flat_params)
            return None, fixed_pars, result

        seeds_flat = self.make_seeds(fixed_pars)
        bounds = self.make_bounds(fixed_pars)

        result = minimize(
            fmin_and_grad,
            seeds_flat,
            bounds=bounds,
            jac=True,
            method="L-BFGS-B",
            tol=self.tol,
            options={"maxls": 50, "maxcor": 50},
        )

        res_dict = restructure_args(result.x, self.parameters, fixed_pars)

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
    def __init__(
        self,
        llh: AbstractLikelihood,
        tol: float = 1e-2,
        strategy: int = 1,
        simplex_prefit: bool = False,
    ):
        super().__init__(
            llh=llh,
        )
        self.tol = tol
        self.strategy = strategy
        self.simplex_prefit = simplex_prefit

    @staticmethod
    def _build_message(minuit):
        """
        Helper function for building a short fit message.
        """
        if minuit.valid:
            message = "Optimization terminated successfully"
            if minuit.accurate:
                message += "."
            else:
                message += ", but uncertainties are unrealiable."
        else:
            message = "Optimization failed."
            fmin = minuit.fmin
            if fmin is not None:
                if fmin.has_reached_call_limit:
                    message += " Call limit was reached."
                if fmin.is_above_max_edm:
                    message += " Estimated distance to minimum too large."

        return message

    def minimize(
        self,
        obs: Dict,
        dataset: Dict,
        fixed_pars: Optional[Dict[str, float]] = None,
    ) -> Tuple[Any, Dict[str, Any], Array]:
        if fixed_pars is None:
            fixed_pars = {}

        wrapped_lh = WrappedLLH(self.llh, obs, dataset, fixed_pars)

        func = wrap_func_np_cache(backend.compile(wrapped_lh))
        grad = wrap_func_np_cache(backend.compile(backend.grad(wrapped_lh)))

        names = [par_name for par_name in self.parameters if par_name not in fixed_pars]

        seeds = self.make_seeds(fixed_pars)

        minuit = iminuit.Minuit(func, seeds, grad=grad, name=names)  # type: ignore
        bounds = self.make_bounds(fixed_pars)

        bound_list = [[lb, ub] for lb, ub in zip(bounds.lb, bounds.ub)]
        minuit.errordef = minuit.LIKELIHOOD
        minuit.limits = bound_list
        minuit.strategy = self.strategy
        minuit.tol = self.tol
        minuit.print_level = 0

        if self.simplex_prefit:
            minuit.simplex().migrad()
        else:
            minuit.migrad()

        minimizer_info = {
            "success": minuit.valid,
            "message": self._build_message(minuit),
            "nfev": minuit.nfcn,
            "njev": minuit.ngrad,
            "hess_inv": minuit.covariance,
        }

        res_dict = restructure_args(
            minuit.values,
            self.parameters,
            fixed_pars,
        )

        fun = backend.array(cast(float, minuit.fval))
        print("best-fit llh: ", fun)
        print("----------------")
        print("best-fit parameters:")
        for key, val in res_dict.items():
            print(f"{key}: {val}")
        print("----------------")

        if fun is None:
            raise RuntimeError("Minimization failed, no function value returned.")

        return minimizer_info, res_dict, fun
