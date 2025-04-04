from typing import Callable, Dict, Any, Tuple, List
from scipy.optimize import minimize, Bounds
from .likelihood import AbstractLikelihood, Prior, GaussianPrior
from .backend import backend
import iminuit


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
    def __init__(self, likelihood: AbstractLikelihood,
                 obs: Dict,
                 datasets: Dict,
                 fixed_params: Dict,
                 prior: List[Prior]):
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
        exp_vars = self.likelihood.get_analysis().exposed_variables()
        restructured_args = restructure_args(flat_params,
                                             exp_vars,
                                             self.fixed_params)
        binned_llh = -self.likelihood.llh(self.obs,
                                          self.datasets,
                                          restructured_args)
        for p in self.prior:
            binned_llh += p.evaluate(restructured_args)
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
        Seeds: Initial guesses for the optimization parameters.
        Priors (Dict): Prior distributions for the parameters.
        fixed_pars (Dict): Parameters to keep fixed during optimization.
    """
    llh = None
    par_idx_map = None

    def __init__(self,
                 llh: AbstractLikelihood,
                 obs: Dict,
                 dataset: Dict,
                 exposed_vars: Dict[str, List[str]],
                 bounds: Dict[str, Dict[str, Tuple[float, float]]],
                 Seeds: Dict[str, Dict[str, float]],
                 Priors: Dict[str, Dict[str, Tuple[float, float]]] = {},
                 fixed_pars: Dict[str, Dict[str, float]] = {}):
        raise NotImplementedError("Subclasses should implement this method.")

    def flat_index_mapping(self,
                           exposed_vars: Dict[str, List[str]],
                           fixed_pars: Dict[str, Dict[str, float]],
                           bounds: Dict[str, Dict[str, Tuple[float, float]]],
                           Seeds: Dict[str, Dict[str, float]]
                           ):
        par_idx_map = flat_index_dict_mapping(exposed_vars, fixed_pars)
        bounds_lower = []
        bounds_upper = []
        seeds = []
        for factor, pars in par_idx_map.items():
            for par in pars:
                lower, upper = bounds[factor][par]
                bounds_lower.append(lower)
                bounds_upper.append(upper)
                seeds.append(Seeds[factor][par])
        return par_idx_map, seeds, bounds_lower, bounds_upper

    def get_prior_list(self,
                       Priors: Dict[str, Dict[str, Tuple[float, float]]]):
        prior_objects = []
        for factor, pars in Priors.items():
            prior_objects.append(GaussianPrior(factor, pars))
        return prior_objects

    def minimize(self):
        raise NotImplementedError("Subclasses should implement this method.")


class Scipy_Minimizer(AbstractMinimizer):
    def __init__(self,
                 llh: AbstractLikelihood,
                 obs: dict,
                 dataset: dict,
                 exposed_vars: Dict[str, List[str]],
                 bounds: Dict[str, Dict[str, Tuple[float, float]]],
                 Seeds: Dict[str, Dict[str, float]],
                 Priors: Dict[str, Dict[str, Tuple[float, float]]] = {},
                 fixed_pars: Dict[str, Dict[str, float]] = {}):
        prior_objects = self.get_prior_list(Priors)
        wrapped_lh = WrappedLLH(llh, obs, dataset, fixed_pars, prior_objects)
        self.fmin_and_grad = backend.func_and_grad(wrapped_lh)
        (self.par_idx_map,
         self.seeds,
         bounds_lower,
         bounds_upper) = self.flat_index_mapping(
            exposed_vars,
            fixed_pars,
            bounds,
            Seeds)
        self.bounds = Bounds(bounds_lower, bounds_upper)

    def minimize(self):
        result = minimize(
            self.fmin_and_grad,
            self.seeds,
            bounds=self.bounds,
            jac=True,
            method="L-BFGS-B",
            tol=1e-8,
            options={"maxls": 50, }
            )
        return result


class Minuit_Minimizer(AbstractMinimizer):
    def __init__(self,
                 llh: AbstractLikelihood,
                 obs: dict,
                 dataset: dict,
                 exposed_vars: Dict[str, List[str]],
                 bounds: Dict[str, Dict[str, Tuple[float, float]]],
                 Seeds: Dict[str, Dict[str, float]],
                 Priors: Dict[str, Dict[str, Tuple[float, float]]] = {},
                 fixed_pars: Dict[str, Dict[str, float]] = {}):
        prior_objects = self.get_prior_list(Priors)
        wrapped_lh = WrappedLLH(llh,
                                obs,
                                dataset,
                                fixed_pars,
                                prior_objects)
        self.func = backend.compile(wrapped_lh)
        self.grad = backend.compile(backend.grad(wrapped_lh))
        print("Compiled Likelihood function and gradient")
        (self.par_idx_map,
         self.seeds,
         bounds_lower,
         bounds_upper) = self.flat_index_mapping(
            exposed_vars,
            fixed_pars,
            bounds,
            Seeds
            )
        print("Setup Minimizer")
        self.minuit = iminuit.Minuit(
            self.func,
            self.seeds,
            grad=self.grad,
            name=self.par_names
        )
        N_par = len(self.seeds)
        bound_list = [[bounds_lower[i], bounds_upper[i]] for i in range(N_par)]
        self.minuit.errordef = self.minuit.LIKELIHOOD
        self.minuit.limits = bound_list
        self.minuit.strategy = 1
        self.minuit.tol = 1e-2
        self.minuit.print_level = 0
        self.par_names = []
        for factor, pars in self.par_idx_map.items():
            for par in pars:
                self.par_names.append(factor + "_" + par)

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
        print("Starting Minimization")
        self.minuit.migrad()
        print("Finished Minimization")
        minimizer_info = {
            'success': self.minuit.valid,
            'message': self._build_message(),
            'nfev': self.minuit.nfcn,
            'njev': self.minuit.ngrad,
            'hess_inv': self.minuit.covariance,
        }
        res_dict = {
            var: self.minuit.values[var]
            for var in self.par_names
        }
        fun = self.minuit.fval
        print("best-fit llh: ", fun)
        print("----------------")
        print("best-fit parameters:")
        for factor, pars in self.par_idx_map.items():
            print("")
            print(factor)
            for par, val in pars.items():
                print(f"{par}: {res_dict[factor+'_'+par]:0.3f}")
        print("----------------")

        return minimizer_info, res_dict, fun
