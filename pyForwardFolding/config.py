from typing import Dict

import yaml

from .analysis import Analysis
from .binned_expectation import BinnedExpectation
from .binning import AbstractBinning
from .factor import AbstractBinnedFactor, AbstractFactor, AbstractUnbinnedFactor
from .model import Model
from .model_component import ModelComponent


def _load_config(path: str) -> Dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def _build_factors(conf: Dict) -> Dict[str, AbstractFactor]:
    factors = [AbstractUnbinnedFactor.construct_from(f) for f in conf["factors"]]
    return {f.name: f for f in factors}


def _build_components(
    conf: Dict, factors: Dict[str, AbstractFactor]
) -> Dict[str, ModelComponent]:
    components = [
        ModelComponent(
            c["name"],
            [factors[fname] for fname in c["factors"]],
        )
        for c in conf["components"]
    ]
    return {c.name: c for c in components}


def _build_models(
    conf: Dict, components: Dict[str, ModelComponent]
) -> Dict[str, Model]:
    models = [
        Model.from_pairs(
            m["name"],
            [(c["baseline_weight"], components[c["name"]]) for c in m["components"]],
        )
        for m in conf["models"]
    ]
    return {m.name: m for m in models}


def analysis_from_config(path: str) -> Analysis:
    """
    Load an analysis configuration from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        Analysis: The constructed analysis object.
    """
    conf = _load_config(path)
    factors = _build_factors(conf)
    components = _build_components(conf, factors)
    models = _build_models(conf, components)

    binned_expectations = {}

    for hist_config in conf["histograms"]:
        binning = AbstractBinning.construct_from(hist_config["binning"])
        lifetime = hist_config.get("lifetime", 1.0)
        hist_factors = [
            AbstractBinnedFactor.construct_from(f, binning)
            for f in hist_config.get("hist_factors", [])
        ]

        dskey_model_name_pairs = hist_config["models"]
        dskey_model_pairs = [
            (dskey, models[model_name]) for model_name, dskey in dskey_model_name_pairs
        ]

        binned_expectations[hist_config["name"]] = BinnedExpectation(
            name=hist_config["name"],
            dskey_model_pairs=dskey_model_pairs,
            binning=binning,
            binned_factors=hist_factors,
            lifetime=lifetime,
        )

    return Analysis(binned_expectations)


def models_from_config(path: str) -> Dict[str, Model]:
    """
    Load models per dataset from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: model for each dataset.
    """
    conf = _load_config(path)
    factors = _build_factors(conf)
    components = _build_components(conf, factors)
    models = _build_models(conf, components)

    output = {}
    for hist in conf["histograms"]:
        output[hist["name"]] = {}
        for model in hist["models"]:
            output[hist["name"]][model[0]] = models[model[0]]

    return output
