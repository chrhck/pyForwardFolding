import pickle
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np

from .backend import backend
from .binning import AbstractBinning


class AbstractFactor:
    """
    Abstract class representing a per-event factor.
    """

    def __init__(self, name:str, param_mapping: Dict[str, str] = None):
        """
        Initialize the factor with a name and parameter mapping.
        Args:
            name (str): Identifier for the factor.
            param_mapping (dict): Dictionary mapping factor parameter names to names in the parameter dictionary.
        """
        self.name = name
        
        self.param_mapping = param_mapping

        self.factor_parameters: List[str] = []
        self.req_vars: List[str] = []

    @property
    def required_variables(self) -> List[str]:
        return self.req_vars

    @property
    def parameter_mapping(self) -> Dict[str, str]:
        if self.param_mapping is None:
            return  {par: par for par in self.factor_parameters}
        return self.param_mapping
    
    @property
    def exposed_parameters(self) -> List[str]:
        return list(self.parameter_mapping.values())
    
    
    def evaluate(
        self,
        input_variables: Dict[str, Union[np.ndarray, float]],
        parameters: Dict[str, Union[np.ndarray, float]],
    ) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def construct_from(cls: Type["AbstractFactor"], config: Dict[str, Any]) -> "AbstractFactor":
        
        factor_type = config.get("type")
        factor_class = FACTORSTR_CLASS_MAPPING.get(factor_type)

        if factor_class is None:
            raise ValueError(f"Unknown factor type: {factor_type}")

        return factor_class.construct_from(config)

def get_required_variable_values(factor, input_variable_values):
    """
    Extract required variable values for a factor from the input dictionary.

    Args:
        factor (AbstractFactor): The factor requesting variables.
        input_variable_values (dict): Dictionary containing all available input variables.

    Returns:
        dict: Dictionary containing only the required variables for the factor.
    """
    req_vars = factor.required_variables
    return {var: input_variable_values[var] for var in req_vars}


def get_parameter_values(factor, parameter_dict):
    """
    Extract parameter values for a factor from the parameter dictionary.

    Args:
        factor (AbstractFactor): The factor requesting variables.
        parameter_dict (dict): Dictionary mapping parameter names to values.

    Returns:
        dict: Dictionary containing only the exposed variables for the factor.
    """

    par_mapping = factor.parameter_mapping
    parameter_values = {factor_var_name: parameter_dict[par_name] for factor_var_name, par_name in par_mapping.items()}
    return parameter_values


class PowerLawFlux(AbstractFactor):
    """
    Factor that applies a power law flux model.

    Args:
        name (str): Identifier for the factor.
        pivot_energy (float): Reference energy for the power law.
        baseline_norm (float): Baseline normalization factor.
    """

    

    def __init__(self, name:str, pivot_energy, baseline_norm, param_mapping: Dict[str, str]=None):
        
        super().__init__(name, param_mapping)

        self.pivot_energy = pivot_energy
        self.baseline_norm = baseline_norm

        self.factor_parameters: List[str] = ["flux_norm", "spectral_index"]
        self.req_vars: List[str] = ["true_energy"]

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "PowerLawFlux":
        param_mapping = config.get("param_mapping", None)
        return PowerLawFlux(
                name=config["name"],
                pivot_energy=config["pivot_energy"],
                baseline_norm=config["baseline_norm"],
                param_mapping=param_mapping,
        )


    def evaluate(self, input_variables, parameter_values):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameter_values)
        true_energy = input_values["true_energy"]
        flux_norm = exposed_values["flux_norm"]
        spectral_index = exposed_values["spectral_index"]

        return flux_norm * self.baseline_norm * backend.power(true_energy / self.pivot_energy, -spectral_index)


class FluxNorm(AbstractFactor):
    """
    Factor that applies a simple flux normalization.

    Args:
        name (str): Identifier for the factor.
    """

    def __init__(self, name:str, param_mapping: Dict[str, str] = None):
        super().__init__(name, param_mapping)

        self.factor_parameters = ["flux_norm"]
        self.req_vars = []

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "FluxNorm":
        param_mapping = config.get("param_mapping", None)
        return FluxNorm(
                name=config["name"],
                param_mapping=param_mapping,
        )

    def evaluate(self, input_variables, parameter_values):
        exposed_values = get_parameter_values(self, parameter_values)
        flux_norm = exposed_values["flux_norm"]

        return flux_norm


class SnowstormGauss(AbstractFactor):
    """
    Factor that implements a Gaussian reweighting scheme for systematic uncertainty modeling.

    Args:
        name (str): Identifier for the factor.
        sys_gauss_width (float): Width of the Gaussian distribution.
        sys_sim_bounds (tuple): Bounds for the simulated parameter space (min, max).
        req_variable_name (str): Name of the required variable for reweighting.
    """

    def __init__(self, name, sys_gauss_width, sys_sim_bounds, req_variable_name, param_mapping: Dict[str, str] = None):
        super().__init__(name, param_mapping)

        self.sys_gauss_width = sys_gauss_width
        self.sys_sim_bounds = sys_sim_bounds
        self.req_vars = [req_variable_name]
        self.factor_parameters = ["scale"]

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "SnowstormGauss":
        param_mapping = config.get("param_mapping", None)
        return SnowstormGauss(
                name=config["name"],
                sys_gauss_width=config["sys_gauss_width"],
                sys_sim_bounds=tuple(config["sys_sim_bounds"]),
                req_variable_name=config["req_variable_name"],
                param_mapping=param_mapping
        )


    def evaluate(self, input_variables, parameter_values):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameter_values)
        sys_value = exposed_values["scale"]
        sys_par = input_values[self.req_vars[0]]

        return (
            (
                backend.gauss_pdf(sys_par, sys_value, self.sys_gauss_width) /
                backend.gauss_cdf(self.sys_sim_bounds[1], sys_value, self.sys_gauss_width)
            ) /
            backend.uniform_pdf(sys_par, self.sys_sim_bounds[0], self.sys_sim_bounds[1])
        )        


class DeltaGamma(AbstractFactor):
    """
    Factor that applies a delta gamma scaling.

    Args:
        name (str): Identifier for the factor.
        reference_energy (float): Reference energy for scaling.
    """

    def __init__(self, name, param_mapping: Dict[str, str] = None, reference_energy: float = 1.0):
        super().__init__(name, param_mapping)
        self.reference_energy = reference_energy

        self.factor_parameters = ["delta_gamma"]
        self.req_vars = ["true_energy", "median_energy"]

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "DeltaGamma":
        param_mapping = config.get("param_mapping", None)
        return DeltaGamma(
                name=config["name"],
                reference_energy=config["reference_energy"],
                param_mapping=param_mapping,
        )

    def evaluate(self, input_variables, parameter_values):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameter_values)

        delta_gamma = exposed_values["delta_gamma"]
        true_energy = input_values["true_energy"]
        # median_energy = input_values["median_energy"]
        return backend.power(true_energy / self.reference_energy, -delta_gamma)


class ModelInterpolator(AbstractFactor):
    """
    Interpolation between two models.

    Args:
        name (str): Identifier for the factor.
        baseline_weight (str): Name of the baseline weight variable.
        alternative_weight (str): Name of the alternative weight variable.
        param_mapping (dict): Dictionary mapping factor parameter names to names in the parameter dictionary.
    """

    

    def __init__(self, name: str, baseline_weight: Dict[str, str], alternative_weight: Dict[str, str], param_mapping: Dict[str, str] = None):
        super().__init__(name, param_mapping)
        self.base_key = baseline_weight
        self.alt_key = alternative_weight
        self.req_vars = [self.base_key, self.alt_key]
        self.factor_parameters: List[str] = ["lambda_int"]

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "ModelInterpolator":
        param_mapping = config.get("param_mapping", None)
        return ModelInterpolator(
                name=config["name"],
                baseline_weight=config["baseline_weight"],
                alternative_weight=config["alternative_weight"],
                param_mapping=param_mapping,
        )

    def evaluate(self, input_variables, parameters):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameters)
        baseline_weight = input_values[self.base_key]
        alternative_weight = input_values[self.alt_key]
        lambda_int = exposed_values["lambda_int"]


        # If baseline weight is 0 also return 1
        sanitized_baseline_weight = backend.where(
            baseline_weight == 0,
            1,
            baseline_weight)

        log_sanitized_baseline_weight = backend.log(sanitized_baseline_weight)
        log_alternative_weight = backend.log(alternative_weight)


        result = backend.where(
             baseline_weight == 0,
             1,
             (1-lambda_int) + lambda_int* backend.exp(log_alternative_weight - log_sanitized_baseline_weight)
        )
        

        return result


class GradientReweight(AbstractFactor):
    """
    Gradient reweight application. (e.g barr parameters)
    Requires precalculated gradients.

    Args:
        name (str): Identifier for the factor.
        gradient_key_mapping (dict): Dictionary mapping exposed variable names to gradient variable names.
        baseline_weight (str): Name of the baseline weight variable.
    """

    def __init__(self, name: str, gradient_key_mapping: Dict[str, str], baseline_weight: str, param_mapping: Dict[str, str] = None):
        super().__init__(name, param_mapping)
        self.baseline_weight = baseline_weight
        self.grad_key_map = gradient_key_mapping
        self.req_vars = list(self.grad_key_map.values()) + [self.baseline_weight]
        self.factor_parameters = list(self.grad_key_map.keys())

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "GradientReweight":
        param_mapping = config.get("param_mapping", None)
        return GradientReweight(
                name=config["name"],
                gradient_key_mapping=config["gradient_key_mapping"],
                baseline_weight=config["baseline_weight"],
                param_mapping=param_mapping,
        )

    def evaluate(self, input_variables, parameters):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameters)
        baseline = input_values[self.baseline_weight]
        reweight = baseline
        for par in self.factor_parameters:
            par_gradient = input_variables[self.grad_key_map[par]]
            par_value = exposed_values[par]
            reweight += par_value*par_gradient
        return reweight/baseline


class VetoThreshold(AbstractFactor):
    """
    Changes the atm. passing fraction according to a second-order expansion of
    log10(splined_passing_fraction)
    """

   

    def __init__(
            self,
            name,
            threshold_a,
            threshold_b,
            threshold_c,
            rescale_energy,
            anchor_energy,
            param_mapping: Dict[str, str] = None):
        """
        Read fir coefficients as well as anchor energy [GeV]

        Args:
            threshold_: coefficients of second-order expansion
            anchor_energy: energy at which log10(PF) was expanded
            rescale_energy: scale 10**(fit parameter) to energy # TODO
        """
        
        super().__init__(name, param_mapping)

        self.a = threshold_a
        self.b = threshold_b
        self.c = threshold_c
        self.e_rescale = rescale_energy
        self.e_anchor = anchor_energy
        self.req_vars = [self.a, self.b, self.c]
        self.factor_parameters: List[str] = ["e_threshold"]

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "VetoThreshold":
        param_mapping = config.get("param_mapping", None)
        return VetoThreshold(
                name=config["name"],
                threshold_a=config["threshold_a"],
                threshold_b=config["threshold_b"],
                threshold_c=config["threshold_c"],
                rescale_energy=config["rescale_energy"],
                anchor_energy=config["anchor_energy"],
                param_mapping=param_mapping,
        )

    def evaluate(self, input_variables, parameters):

        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameters)
        e_threshold = exposed_values["e_threshold"]
        a = input_values[self.a]
        b = input_values[self.b]
        c = input_values[self.c]
        # parameter value to original energy scale, minus point at which
        # expansion was done
        # parameter itself transformed to log10:
        # e_t is log_10(energy threshold / 100 GeV)
        e = self.e_rescale * backend.exp(
            backend.log(10) * e_threshold
        ) - self.e_anchor
        # i.e. energy threshold in [5 GeV, 3 TeV] = e_t in [-1.301, 1.477]

        # second order expansion from fit coefficients
        log_pf = a + b*e + c*e*e
        # expansion is in log10(passing_fraction)
        reweight = backend.exp(backend.log(10) * log_pf)
        # atm. weights are multiplied by passing fraction
        return reweight

class AbstractBinnedFactor(AbstractFactor):
    """
    Abstract base class for factors that contribute to a binned expectation.
    This class should be inherited by specific implementations of binned factors.
    """

    def __init__(self, name: str, binning:AbstractBinning, param_mapping: Dict[str, str] = None):
        """
        Initialize the AbstractBinnedFactor with a name.

        Args:
            name (str): Identifier for the factor.
        """
        super().__init__(name, param_mapping)
        self.binning = binning


    @classmethod
    def construct_from(cls: Type["AbstractFactor"], config: Dict[str, Any], binning:AbstractBinning) -> "AbstractFactor":
        
        factor_type = config.get("type")
        factor_class = FACTORSTR_CLASS_MAPPING.get(factor_type)


        if factor_class is None:
            raise ValueError(f"Unknown factor type: {factor_type}")

        return factor_class.construct_from(config, binning)

class SnowStormGradient(AbstractBinnedFactor):

    """
    Factor that applies a systematic parameter gradient.
    Is aplied additive to each detector histogram.
    """

    def __init__(
        self,
        name: str,
        binning: AbstractBinning,
        parameters: List[str],
        gradient_names: List[str],
        default: List[float],
        split_values: List[Tuple],
        gradient_pickle: str,
        param_mapping: Dict[str, str] = None

    ):
        """
        Parameters
        ----------
        name : str
            Name of the factor
        binning : AbstractBinning
            Binning object for the factor
        parameters : list
            List of parameter names
        gradient_names : list
            List of gradient dictionary keys for each parameter
        default : list
            List of default parameter values
        split_values : list
            List of split values for each parameter
        gradient_pickle : str
            Path to pickle file containing the gradients

        """
        super().__init__(name, binning, param_mapping)

        self.defaults = default
        self.split_values = split_values
        self.gradient_names = gradient_names

        with open(gradient_pickle, "rb") as f:
            self.gradients = pickle.load(f)

        self.factor_parameters = parameters

        ndim_grads = [len(be)-1 for be in self.gradients["binning"]]

        if list(self.binning.hist_dims) != ndim_grads:
            raise ValueError(
                f"Mismatch between binning dimensions ({self.binning.hist_dims}) and gradient dimensions ({ndim_grads})"
            )
        
        # TODO: check if bin edges are compatible?


    @classmethod
    def construct_from(cls, config: Dict[str, Any], binning:AbstractBinning) -> "SnowStormGradient":
        return SnowStormGradient(
            name=config["name"],
            binning=binning,
            parameters=config["parameters"],
            gradient_names=config["gradient_names"],
            default=config["default"],
            split_values=config["split_values"],
            gradient_pickle=config["gradient_pickle"],
            param_mapping = config.get("param_mapping", None)
        )

    def evaluate(self, input_variables, parameters):
        """
        Evaluate the systematic parameter gradient for the given
        detector configuration and the given exposed variables.

        """
       
        t_gradients = float(
            self.gradients["livetime"]
        )
        exposed_values = get_parameter_values(self, parameters)

        # calculate variation of systematic parameters w.r.t. split
        #   value in order to correctly relate to the gradient dict.
        #   Overall parameter shifts are taken into account here so
        #   that gradients are applied w.r.t. their split value
        #   but the fit parameter itself corresponds to the shifted value
        mu_add = 0
        ssq_add = 0


        for i, sys_par in enumerate(self.exposed_parameters):
            sys_val = exposed_values[sys_par]
            gradient = self.gradients[self.gradient_names[i]]

            mu_add += (sys_val - self.split_values[i]) *\
                gradient["gradient"]
            ssq_add += ((sys_val - self.split_values[i]) *
                        gradient["gradient_error"])**2

        return mu_add/t_gradients, ssq_add/t_gradients**2


class ScaledTemplate(AbstractBinnedFactor):
    def __init__(self, name: str, binning: AbstractBinning, template_file: str, param_mapping: Dict[str, str] = None):

        super().__init__(name, binning, param_mapping)

        with open(template_file, "rb") as f:
            self.template = pickle.load(f)

        self.factor_parameters = ["template_norm"]
 
    @classmethod
    def construct_from(cls, config: Dict[str, Any], binning:AbstractBinning) -> "ScaledTemplate":
        return ScaledTemplate(
            name=config["name"],
            binning=binning,
            template_file=config["template_file"],
            param_mapping = config.get("param_mapping", None)
        )

    def evaluate(self, input_variables, parameters):
        exposed_values = get_parameter_values(self, parameters)
        template_norm = exposed_values["template_norm"]

        # TODO: compare template binning with configured one
        if "template_fluctuation" in self.template:
            template_fluct = (self.template["template_fluctuation"]*template_norm)**2
            template_fluct = template_fluct.reshape(self.binning.hist_dims)
        else:
            template_fluct = None
        return (self.template["template"] * template_norm).reshape(self.binning.hist_dims), template_fluct
    
FACTORSTR_CLASS_MAPPING = {
    "PowerLawFlux": PowerLawFlux,
    "FluxNorm": FluxNorm,
    "SnowstormGauss": SnowstormGauss,
    "DeltaGamma": DeltaGamma,
    "GradientReweight": GradientReweight,
    "ModelInterpolator": ModelInterpolator,
    "VetoThreshold": VetoThreshold,
    "SnowStormGradient": SnowStormGradient,
    "ScaledTemplate": ScaledTemplate,
}