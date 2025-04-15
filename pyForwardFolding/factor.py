from typing import Any, Dict, List, Type, Union

import numpy as np

from .backend import backend


class AbstractFactor:
    """
    Abstract class representing a per-event factor.
    """
    factor_parameters: List[str] = []
    req_vars: List[str] = []

    def __init__(self, name:str, param_mapping: Dict[str, str] = None):
        """
        Initialize the factor with a name and parameter mapping.
        Args:
            name (str): Identifier for the factor.
            param_mapping (dict): Dictionary mapping factor parameter names to names in the parameter dictionary.
        """
        self.name = name

        if param_mapping is None:
            param_mapping = {par: par for par in self.factor_parameters}
        
        self.param_mapping = param_mapping

    @property
    def required_variables(self) -> List[str]:
        return self.req_vars

    @property
    def parameter_mapping(self) -> Dict[str, str]:
        return self.param_mapping
    
    @property
    def exposed_parameters(self) -> List[str]:
        return self.factor_parameters
    
    
    def evaluate(
        self,
        input_variables: Dict[str, Union[np.ndarray, float]],
        parameters: Dict[str, Union[np.ndarray, float]],
    ) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def construct_from(cls: Type["AbstractFactor"], config: Dict[str, Any]) -> "AbstractFactor":
        factor_type = config.get("type")
        param_mapping = config.get("param_mapping", None)
        if factor_type == "PowerLawFlux":
            return PowerLawFlux(
                name=config["name"],
                pivot_energy=config["pivot_energy"],
                baseline_norm=config["baseline_norm"],
                param_mapping=param_mapping,
            )
        elif factor_type == "FluxNorm":
            return FluxNorm(name=config["name"], param_mapping=param_mapping)
        elif factor_type == "SnowstormGauss":
            return SnowstormGauss(
                name=config["name"],
                sys_gauss_width=config["sys_gauss_width"],
                sys_sim_bounds=tuple(config["sys_sim_bounds"]),
                req_variable_name=config["req_variable_name"],
                param_mapping=param_mapping
            )
        elif factor_type == "DeltaGamma":
            return DeltaGamma(
                name=config["name"],
                reference_energy=config["reference_energy"],
                param_mapping=param_mapping
            )
        elif factor_type == "GradientReweight":
            return GradientReweight(
                name=config["name"],
                baseline_weight=config["baseline_weight"],
                gradient_key=config["gradient_key"],
            )
        elif factor_type == "ModelInterpolator":
            return ModelInterpolator(
                name=config["name"],
                base_key=config["base_key"],
                alt_key=config["alt_key"],
            )
        elif factor_type == "SnowStormGradient":
            return SnowStormGradient(
                name=config["name"],
                det_configs=config["det_configs"],
                parameters=config["parameters"],
                default=config["default"],
                split_values=config["split_values"],
                gradient_pickle=config["gradient_pickle"],
                param_in_dict=config["MC_variables"],
            )
        elif factor_type == "ScaledTemplate":
            return ScaledTemplate(
                name=config["name"],
                det_configs=config["det_configs"],
                template_file=config["template_file"],
            )
        elif factor_type == "VetoThreshold":
            return VetoThreshold(
                name=config["name"],
                threshold_a=config["threshold_a"],
                threshold_b=config["threshold_b"],
                threshold_c=config["threshold_c"],
                rescale_energy=config["rescale_energy"],
                anchor_energy=config["anchor_energy"],
            )
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")


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

    factor_parameters: List[str] = ["flux_norm", "spectral_index"]
    req_vars: List[str] = ["true_energy"]

    def __init__(self, name:str, pivot_energy, baseline_norm, param_mapping: Dict[str, str]=None):
        
        super().__init__(name, param_mapping)

        self.pivot_energy = pivot_energy
        self.baseline_norm = baseline_norm


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

    factor_parameters: List[str] = ["flux_norm"]
    required_variables: List[str] = []

    def __init__(self, name:str, param_mapping: Dict[str, str] = None):
        super().__init__(name, param_mapping)

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

    factor_parameters: List[str] = ["scale"]

    def __init__(self, name, sys_gauss_width, sys_sim_bounds, req_variable_name, param_mapping: Dict[str, str] = None):
        super().__init__(name, param_mapping)

        self.sys_gauss_width = sys_gauss_width
        self.sys_sim_bounds = sys_sim_bounds
        self.req_vars = [req_variable_name]


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

    factor_parameters: List[str] = ["delta_gamma"]
    req_vars: List[str] = ["true_energy"]

    def __init__(self, name, reference_energy, param_mapping: Dict[str, str] = None):
        super().__init__(name, param_mapping)
        self.reference_energy = reference_energy


    def evaluate(self, input_variables, parameter_values):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameter_values)

        delta_gamma = exposed_values["delta_gamma"]
        true_energy = input_values["true_energy"]
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

    factor_parameters: List[str] = ["lambda_int"]

    def __init__(self, name: str, baseline_weight: Dict[str, str], alternative_weight: Dict[str, str], param_mapping: Dict[str, str] = None):
        super().__init__(name, param_mapping)
        self.base_key = baseline_weight
        self.alt_key = alternative_weight
        self.req_vars = [self.base_key, self.alt_key]

    def evaluate(self, input_variables, parameters):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameters)
        baseline_weight = input_values[self.base_key]
        alternative_weight = input_values[self.alt_key]
        lambda_int = exposed_values["lambda_int"]
        return lambda_int + (1-lambda_int)*alternative_weight/baseline_weight


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
        self.req_vars = list(self.gradient_key.values()) + [self.baseline_weight]
        self.factor_parameters = list(self.gradient_key.keys())

    def evaluate(self, input_variables, parameters):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_parameter_values(self, parameters)
        baseline = input_values[self.baseline_weight]
        reweight = baseline
        for par in self.factor_parameters:
            par_gradient = input_variables[self.gradient_key[par]]
            par_value = exposed_values[par]
            reweight += par_value*par_gradient
        return reweight/baseline


class VetoThreshold(AbstractFactor):
    """
    Changes the atm. passing fraction according to a second-order expansion of
    log10(splined_passing_fraction)
    """

    factor_parameters: List[str] = ["e_threshold"]

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
