from .backend import backend
from typing import List, Dict, Union, Any, Type
import numpy as np


class AbstractFactor:
    """
    Abstract class representing a factor in a forward folding model.
    """
    def required_variables(self) -> List[str]:
        raise NotImplementedError

    def exposed_variables(self) -> List[str]:
        raise NotImplementedError

    def evaluate(
        self,
        output: np.ndarray,
        input_variables: Dict[str, Union[np.ndarray, float]],
        exposed_variables: Dict[str, Union[np.ndarray, float]],
    ) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def construct_from(cls: Type["AbstractFactor"], config: Dict[str, Any]) -> "AbstractFactor":
        factor_type = config.get("type")
        if factor_type == "EnergyScaler":
            return EnergyScaler(name=config["name"])
        elif factor_type == "PowerLawFlux":
            return PowerLawFlux(
                name=config["name"],
                pivot_energy=config["pivot_energy"],
                baseline_norm=config["baseline_norm"],
            )
        elif factor_type == "FluxNorm":
            return FluxNorm(name=config["name"])
        elif factor_type == "SnowstormGauss":
            return SnowstormGauss(
                name=config["name"],
                sys_gauss_width=config["sys_gauss_width"],
                sys_sim_bounds=tuple(config["sys_sim_bounds"]),
                req_variable_name=config["req_variable_name"],
            )
        elif factor_type == "DeltaGamma":
            return DeltaGamma(
                name=config["name"],
                reference_energy=config["reference_energy"],
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
    req_vars = factor.required_variables()
    return {var: input_variable_values[var] for var in req_vars}


def get_exposed_variable_values(factor, exposed_variable_values):
    """
    Extract exposed variable values for a factor from the nested dictionary of exposed variables.

    Args:
        factor (AbstractFactor): The factor requesting variables.
        exposed_variable_values (dict): Dictionary mapping factor names to their exposed variables.

    Returns:
        dict: Dictionary containing only the exposed variables for the factor.
    """
    this_exposed_variable_values = exposed_variable_values[factor.name]
    exp_vars = factor.exposed_variables()
    return {var: this_exposed_variable_values[var] for var in exp_vars}


class EnergyScaler(AbstractFactor):
    """
    Factor that scales output based on true energy.

    Args:
        name (str): Identifier for the factor.
    """
    def __init__(self, name: str):
        self.name = name

    def required_variables(self) -> List[str]:
        return ["true_energy"]

    def exposed_variables(self) -> List[str]:
        return ["energy_scale"]

    def evaluate(
        self,
        output: np.ndarray,
        input_variables: Dict[str, Union[np.ndarray, float]],
        exposed_variables: Dict[str, Union[np.ndarray, float]],
    ) -> np.ndarray:
        input_values = input_variables["true_energy"]
        energy_scale = exposed_variables["energy_scale"]
        output *= (1 + input_values / 1E3 * energy_scale)
        return output


class PowerLawFlux(AbstractFactor):
    """
    Factor that applies a power law flux model.

    Args:
        name (str): Identifier for the factor.
        pivot_energy (float): Reference energy for the power law.
        baseline_norm (float): Baseline normalization factor.
    """
    def __init__(self, name, pivot_energy, baseline_norm):
        self.name = name
        self.pivot_energy = pivot_energy
        self.baseline_norm = baseline_norm

    def required_variables(self):
        return ["true_energy"]

    def exposed_variables(self):
        return ["flux_norm", "spectral_index"]

    def evaluate(self, output, input_variables, exposed_variables):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_exposed_variable_values(self, exposed_variables)
        true_energy = input_values["true_energy"]
        flux_norm = exposed_values["flux_norm"]
        spectral_index = exposed_values["spectral_index"]

        output *= backend.multiply(
            flux_norm * self.baseline_norm,
            backend.power(true_energy / self.pivot_energy, -spectral_index)
        )
        return output


class FluxNorm(AbstractFactor):
    """
    Factor that applies a simple flux normalization.

    Args:
        name (str): Identifier for the factor.
    """
    def __init__(self, name):
        self.name = name

    def required_variables(self):
        return []

    def exposed_variables(self):
        return ["flux_norm"]

    def evaluate(self, output, input_variables, exposed_variables):
        exposed_values = get_exposed_variable_values(self, exposed_variables)
        flux_norm = exposed_values["flux_norm"]

        output *= backend.multiply(flux_norm, output)
        return output


class SnowstormGauss(AbstractFactor):
    """
    Factor that implements a Gaussian reweighting scheme for systematic uncertainty modeling.

    Args:
        name (str): Identifier for the factor.
        sys_gauss_width (float): Width of the Gaussian distribution.
        sys_sim_bounds (tuple): Bounds for the simulated parameter space (min, max).
        req_variable_name (str): Name of the required variable for reweighting.
    """
    def __init__(self, name, sys_gauss_width, sys_sim_bounds, req_variable_name):
        self.name = name
        self.sys_gauss_width = sys_gauss_width
        self.sys_sim_bounds = sys_sim_bounds
        self.req_variable_name = req_variable_name

    def required_variables(self):
        return [self.req_variable_name]

    def exposed_variables(self):
        return ["sys_value"]

    def evaluate(self, output, input_variables, exposed_variables):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_exposed_variable_values(self, exposed_variables)
        sys_value = exposed_values["sys_value"]
        sys_par = input_values[self.req_variable_name]

        output *= backend.multiply(
            backend.gauss_pdf(sys_par, sys_value, self.sys_gauss_width) /
            backend.gauss_cdf(self.sys_sim_bounds[1], sys_value, self.sys_gauss_width),
            1. / backend.uniform_pdf(sys_par, self.sys_sim_bounds[0], self.sys_sim_bounds[1])
        )
        return output


class DeltaGamma(AbstractFactor):
    """
    Factor that applies a delta gamma scaling.

    Args:
        name (str): Identifier for the factor.
        reference_energy (float): Reference energy for scaling.
    """
    def __init__(self, name, reference_energy):
        self.name = name
        self.reference_energy = reference_energy

    def required_variables(self):
        return ["true_energy"]

    def exposed_variables(self):
        return ["delta_gamma"]

    def evaluate(self, output, input_variables, exposed_variables):
        input_values = get_required_variable_values(self, input_variables)
        exposed_values = get_exposed_variable_values(self, exposed_variables)

        delta_gamma = exposed_values["delta_gamma"]
        true_energy = input_values["true_energy"]
        output *= backend.power(true_energy / self.reference_energy, -delta_gamma)
        return output