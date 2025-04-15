from typing import Any, Dict, List, Tuple, Type

from .factor import AbstractFactor, get_parameter_values


class AbstractBinnedFactor(AbstractFactor):
    """
    Abstract base class for factors that contribute to a binned expectation.
    This class should be inherited by specific implementations of binned factors.
    """

    def __init__(self, name: str):
        """
        Initialize the AbstractBinnedFactor with a name.

        Args:
            name (str): Identifier for the factor.
        """
        super().__init__(name)

    @classmethod
    def construct_from(cls: Type["AbstractBinnedFactor"], config: Dict[str, Any]) -> "AbstractBinnedFactor":
        factor_type = config.get("type")
        param_mapping = config.get("param_mapping", None)
        if factor_type == "SnowStormGradient":
            return SnowStormGradient(
                name=config["name"],
                det_configs=config["det_configs"],
                parameters=config["parameters"],
                default=config["default"],
                split_values=config["split_values"],
                gradient_pickle=config["gradient_pickle"],
                param_in_dict=config["MC_variables"],
                param_mapping=param_mapping,
            )
        elif factor_type == "ScaledTemplate":
            return ScaledTemplate(
                name=config["name"],
                det_configs=config["det_configs"],
                template_file=config["template_file"],
                param_mapping=param_mapping,
            )
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")


class SnowStormGradient(AbstractBinnedFactor):

    """
    Factor that applies a systematic parameter gradient.
    Is aaplied additive to each detector histogram.
    """

    def __init__(
        self,
        name: str,
        parameters: List[str],
        default: List[float],
        split_values: List[Tuple],
        gradient_pickle: str,
        param_in_dict: List[str],
        det_configs: Dict[str, str],
    ):
        """
        Parameters
        ----------
        name : str
            Name of the factor
        parameters : list
            List of parameter names
        default : list
            List of default parameter values
        split_values : list
            List of split values for each parameter
        gradient_pickle : str
            Path to pickle file containing the gradients
        param_in_dict : list
            List of dictionary keys for each parameter
        """
        self.name = name
        self.defaults = default
        self.split_values = split_values
        self.param_in_dict = param_in_dict
        with open(gradient_pickle, "rb") as f:
            self.gradient_dict = pickle.load(f)
        self.param_names = parameters
        self.det_configs = det_configs
        self.bin_edges = {}
        for det_c in self.det_configs:
            if det_configs[det_c] in self.gradient_dict:
                binnings = self.gradient_dict[det_configs[det_c]]['binning']
                self.bin_edges[det_c] = []
                self.bin_edges[det_c].append(binnings[0])
                self.bin_edges[det_c].append(binnings[1])

    def exposed_variables(self) -> List[str]:
        return self.param_names

    def evaluate(self, input_variables, parameters, calling_key):
        """
        Evaluate the systematic parameter gradient for the given
        detector configuration and the given exposed variables.

        Parameters
        ----------
        parameters : dict
            Dictionary of exposed variables
        det_conf : str
            Detector configuration for which to evaluate the gradient
        """
        det_conf = self.det_configs[calling_key]
        t_gradients = float(
            self.gradient_dict[det_conf]["settings"]['config'][det_conf]['livetime']
        )
        exposed_values = get_parameter_values(self, parameters)
        gradients = self.gradient_dict[det_conf]
        # calculate variation of systematic parameters w.r.t. split
        #   value in order to correctly relate to the gradient dict.
        #   Overall parameter shifts are taken into account here so
        #   that gradients are applied w.r.t. their split value
        #   but the fit parameter itself corresponds to the shifted value
        mu_add = 0
        ssq_add = 0
        for i, par in enumerate(exposed_values):
            # default case: add (first) variation of
            # p/params_in_dict[i] to dict
            # print(self.gradient_dict[det_conf].keys())
            value = exposed_values[par]
            gradients = self.gradient_dict[det_conf][self.param_in_dict[i]]
            mu_add += (value - self.split_values[i]) *\
                gradients["gradient"].flatten()
            ssq_add += ((value - self.split_values[i]) *
                        gradients["gradient_error"].flatten())**2

        return mu_add/t_gradients, ssq_add/t_gradients**2


class ScaledTemplate(AbstractBinnedFactor):
    def __init__(self, name: str, template_file: str, det_configs: Dict[str, str]):
        import pickle
        self.name = name
        with open(template_file, "rb") as f:
            self.template = pickle.load(f)
        self.det_configs = det_configs
        self.bin_edges = {}
        for det_c in self.det_configs:
            if det_configs[det_c] in self.template:
                self.bin_edges[det_c] = []
                self.bin_edges[det_c].append(self.template[det_configs[det_c]]['energy_bins'])
                self.bin_edges[det_c].append(self.template[det_configs[det_c]]['zenith_bins'])

    def exposed_variables(self) -> List[str]:
        return ["flux_norm"]

    def required_variables(self) -> List[str]:
        return []

    def evaluate(self, input_variables, exposed_variables, calling_key):
        # input_values = get_required_variable_values(self, input_variables)
        # check whether the loaded file has detector configs as "top level" keys
        det_conf = self.det_configs[calling_key]
        if det_conf in self.template:
            # in case the template files contains a tempalte for more than one
            # detector config, access the template_dict for the current on
            template_dict = self.template[det_conf]
        else:
            # default case: this flux component/tempalte is for single detector
            # config only
            template_dict = self.template
        exposed_values = get_parameter_values(self, exposed_variables)
        flux_norm = exposed_values["flux_norm"]

        # TODO: compare template binning with configured one
        if "template_fluctuation" in template_dict:
            template_fluct = (template_dict["template_fluctuation"]*flux_norm)**2
        else:
            template_fluct = None
        return template_dict["template"] * flux_norm, template_fluct