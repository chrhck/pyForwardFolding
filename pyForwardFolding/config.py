from __future__ import annotations

from typing import Dict

import yaml
import os

from .analysis import Analysis
from .binned_expectation import BinnedExpectation
from .binning import AbstractBinning
from .factor import AbstractBinnedFactor, AbstractUnbinnedFactor
from .model import Model
from .model_component import ModelComponent

from .backend import backend


def _load_config(path: str) -> Dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def _build_factors(conf: Dict) -> Dict[str, AbstractUnbinnedFactor]:
    factors = [AbstractUnbinnedFactor.construct_from(f) for f in conf["factors"]]
    return {f.name: f for f in factors}


def _build_components(
    conf: Dict, factors: Dict[str, AbstractUnbinnedFactor]
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


def models_from_config(path: str) -> Dict[str, Dict[str, Model]]:
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

    output: Dict[str, Dict[str, Model]] = {}
    for hist in conf["histograms"]:
        output[hist["name"]] = {}
        for model in hist["models"]:
            output[hist["name"]][model[0]] = models[model[0]]

    return output

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Loads a Pandas DataFrame from a given file path.
    Automatically detects the file format based on its extension.

    Supported formats: CSV, Parquet, Feather, HDF5, Excel, Pickle.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = os.path.splitext(path)[-1].lower()

    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".parquet"]:
        return pd.read_parquet(path)
    elif ext in [".feather", ".ft"]:
        return pd.read_feather(path)
    elif ext in [".h5", ".hdf", ".hdf5"]:
        return pd.read_hdf(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}' for file: {path}. "
            "Supported formats: CSV, Parquet, Feather, HDF5, Excel, Pickle."
        )

def dataset_from_config(path: str) -> Dict[str, Dict[str, float]]:
    """
    Creates a dataset from a yaml config.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: dataset to be used as analysis input.
    """

    conf = _load_config(path)
    dataset = {}

    for subconf in conf["datasets"]:
        df = load_dataframe(subconf["path"])
        subdataset = {}
        for out_key, entry in subconf["param_mapping"].items():
            in_key = entry["df_key"]
            vals = df[in_key]
            trafo = entry.get("transform")
            if trafo:
                vals = getattr(backend, trafo)(vals)
            subdataset[out_key] = backend.asarray(vals)

        if "median_energy" in subconf:
            energies = subconf["median_energy"]["energy_key"]
            weights = subconf["median_energy"]["weight"]
            median_energy = backend.weighted_median(df[energies], df[weights])
            subdataset["median_energy"] = backend.asarray([median_energy])
        dataset[subconf["name"]] = subdataset
    
    return dataset