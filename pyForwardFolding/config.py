import yaml

from .analysis import Analysis
from .binned_expectation import BinnedExpectation
from .binning import AbstractBinning
from .factor import AbstractBinnedFactor, AbstractFactor
from .model import Model
from .model_component import ModelComponent


def _load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def _build_factors(conf: dict) -> dict:
    factors = [AbstractFactor.construct_from(f) for f in conf["factors"]]
    return {f.name: f for f in factors}


def _build_components(conf: dict, factors: dict) -> dict:
    components = [
        ModelComponent(
            c["name"],
            [factors[fname] for fname in c["factors"]],
        )
        for c in conf["components"]
    ]
    return {c.name: c for c in components}


def _build_models(conf: dict, components: dict) -> dict:
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

    for dataset in conf["datasets"]:
        binning = AbstractBinning.construct_from(dataset["binning"])
        lifetime = dataset.get("lifetime", 1.0)
        hist_factors = [
            AbstractBinnedFactor.construct_from(f, binning)
            for f in dataset.get("hist_factors", [])
        ]

        binned_expectations[dataset["name"]] = BinnedExpectation(
            name=dataset["name"],
            model=models[dataset["model"]],
            binning=binning,
            binned_factors=hist_factors,
            lifetime=lifetime,
        )

    return Analysis(binned_expectations)


def models_from_config(path: str) -> dict:
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

    return {
        dataset["name"]: models[dataset["model"]]
        for dataset in conf["datasets"]
    }