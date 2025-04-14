from .backend import backend
from typing import List, Dict, Union, Any, Type
import numpy as np


class AbstractFactor:
    """
    Abstract class representing a per-event factor.
    """
    factor_parameters: List[str] = []

    def __init__(self, name:str, param_mapping: Dict[str, str] = None):
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
    
    
    def evaluate(
        self,
        output: np.ndarray,
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
