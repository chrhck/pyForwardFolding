__version__ = "0.1.0"

# ruff: noqa: I001, E402

import jax 
jax.config.update("jax_enable_x64", True)

import logging
logging.warning("Set JAX to use 64-bit precision")

from . import (
    analysis,
    backend,
    binned_expectation,
    binning,
    config,
    factor,
    likelihood,
    minimizer,
    model,
    model_component,
    statistics,
)

__all__ = [
    "analysis",
    "backend",
    "binned_expectation",
    "binning",
    "config",
    "factor",
    "likelihood",
    "minimizer",
    "model",
    "model_component",
    "statistics",
]
