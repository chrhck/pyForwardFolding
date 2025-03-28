from typing import Dict, Any
import yaml
from .factor import AbstractFactor
from .model_component import ModelComponent
from .model import Model
from .binning import AbstractBinning
from .binned_expectation import BinnedExpectation
from .analysis import Analysis


def analysis_from_config(path: str) -> Analysis:
    """
    Load an analysis configuration from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        Analysis: The constructed analysis object.
    """
    with open(path, "r") as file:
        conf = yaml.safe_load(file)

    factors = [
        AbstractFactor.construct_from(factor_conf)
        for factor_conf in conf["factors"]
    ]
    factors_name_mapping = {f.name: f for f in factors}

    components = [
        ModelComponent(
            c["name"],
            [factors_name_mapping[factor_name] for factor_name in c["factors"]],
        )
        for c in conf["components"]
    ]
    components_name_mapping = {c.name: c for c in components}

    model_conf = conf["model"]
    model = Model(
        model_conf["name"],
        [
            (c["baseline_weight"], components_name_mapping[c["name"]])
            for c in model_conf["components"]
        ],
    )

    dset_config = conf["datasets"]
    binned_expectations = {}

    for dataset in dset_config:
        binning = AbstractBinning.construct_from(dataset["binning"])
        binned_expectations[dataset["name"]] = BinnedExpectation(model, binning)

    ana = Analysis(binned_expectations)
    return ana