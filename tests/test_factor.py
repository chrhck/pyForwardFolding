import math

import jax.numpy as jnp
import pytest

from pyForwardFolding.backend import Backend, JAXBackend
from pyForwardFolding.factor import (
    DeltaGamma,
    EnergyScaler,
    FluxNorm,
    PowerLawFlux,
    SnowstormGauss,
)


# Example: Add a mock backend for testing purposes
class MockBackend(Backend):
    def array(self, data):
        return data

    def multiply(self, a, b):
        return a * b

    def power(self, a, b):
        return a ** b

    def exp(self, a):
        return math.exp(a)

    def sqrt(self, a):
        return math.sqrt(a)

    def erf(self, a):
        return 0  # Simplified for testing

    def gauss_pdf(self, x, mu, sigma):
        return 1.0  # Simplified for testing

    def gauss_cdf(self, x, mu, sigma):
        return 0.5  # Simplified for testing

    def uniform_pdf(self, x, lo, hi):
        return 1.0  # Simplified for testing


# Parameterize tests to use different backends
@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_energy_scaler(backend):
    factor = EnergyScaler(name="energy_scaler")
    input_variables = {"true_energy": 1000.0}
    exposed_variables = {"energy_scaler": {"energy_scale": 0.1}}
    output = backend.array([1.0, 2.0, 3.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([1.1, 2.2, 3.3])  # Scaled by (1 + 1000 / 1E3 * 0.1)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_energy_scaler_edge_cases(backend):
    factor = EnergyScaler(name="energy_scaler")
    input_variables = {"true_energy": 0.0}  # Edge case: zero energy
    exposed_variables = {"energy_scaler": {"energy_scale": 0.1}}
    output = backend.array([1.0, 2.0, 3.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([1.0, 2.0, 3.0])  # No scaling since true_energy is 0
    assert jnp.allclose(result, expected)

    input_variables = {"true_energy": -1000.0}  # Edge case: negative energy
    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([0.9, 1.8, 2.7])  # Scaled by (1 - 1000 / 1E3 * 0.1)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_power_law_flux(backend):
    factor = PowerLawFlux(name="power_law", pivot_energy=1.0, baseline_norm=1.0)
    input_variables = {"true_energy": 10.0}
    exposed_variables = {"power_law": {"flux_norm": 2.0, "spectral_index": 1.0}}
    output = backend.array([1.0, 1.0, 1.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([0.2, 0.2, 0.2])  # flux_norm * (true_energy / pivot_energy)^(-spectral_index)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_power_law_flux_edge_cases(backend):
    factor = PowerLawFlux(name="power_law", pivot_energy=1.0, baseline_norm=1.0)
    input_variables = {"true_energy": 0.0}  # Edge case: zero energy
    exposed_variables = {"power_law": {"flux_norm": 2.0, "spectral_index": 1.0}}
    output = backend.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError):  # Expect error due to division by zero
        factor.evaluate(output, input_variables, exposed_variables)

    input_variables = {"true_energy": 1.0}  # Edge case: pivot energy
    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([2.0, 2.0, 2.0])  # flux_norm * (1 / 1)^(-1)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_flux_norm(backend):
    factor = FluxNorm(name="flux_norm")
    input_variables = {}
    exposed_variables = {"flux_norm": {"flux_norm": 2.0}}
    output = backend.array([1.0, 2.0, 3.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([2.0, 4.0, 6.0])  # Scaled by flux_norm
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_flux_norm_edge_cases(backend):
    factor = FluxNorm(name="flux_norm")
    input_variables = {}
    exposed_variables = {"flux_norm": {"flux_norm": 0.0}}  # Edge case: zero norm
    output = backend.array([1.0, 2.0, 3.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([0.0, 0.0, 0.0])  # All values scaled to zero
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_snowstorm_gauss(backend):
    factor = SnowstormGauss(
        name="snowstorm",
        sys_gauss_width=0.1,
        sys_sim_bounds=(-0.5, 0.5),
        req_variable_name="dom_eff",
    )
    input_variables = {"dom_eff": 0.0}
    exposed_variables = {"snowstorm": {"sys_value": 0.1}}
    output = backend.array([1.0, 1.0, 1.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    # Since the exact Gaussian calculation is complex, we can test for non-zero scaling
    assert jnp.all(result > 0)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_snowstorm_gauss_edge_cases(backend):
    factor = SnowstormGauss(
        name="snowstorm",
        sys_gauss_width=0.0,  # Edge case: zero width
        sys_sim_bounds=(-0.5, 0.5),
        req_variable_name="dom_eff",
    )
    input_variables = {"dom_eff": 0.0}
    exposed_variables = {"snowstorm": {"sys_value": 0.1}}
    output = backend.array([1.0, 1.0, 1.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    assert jnp.all(result == 1.0)  # No scaling since width is zero


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_delta_gamma(backend):
    factor = DeltaGamma(name="delta_gamma", reference_energy=1.0)
    input_variables = {"true_energy": 10.0}
    exposed_variables = {"delta_gamma": {"delta_gamma": 1.0}}
    output = backend.array([1.0, 1.0, 1.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([0.1, 0.1, 0.1])  # (true_energy / reference_energy)^(-delta_gamma)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_delta_gamma_edge_cases(backend):
    factor = DeltaGamma(name="delta_gamma", reference_energy=1.0)
    input_variables = {"true_energy": 0.0}  # Edge case: zero energy
    exposed_variables = {"delta_gamma": {"delta_gamma": 1.0}}
    output = backend.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError):  # Expect error due to division by zero
        factor.evaluate(output, input_variables, exposed_variables)

    input_variables = {"true_energy": 1.0}  # Edge case: reference energy
    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([1.0, 1.0, 1.0])  # (1 / 1)^(-1)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_backend_operations(backend):
    # Test backend operations to ensure correctness
    assert backend.multiply(2, 3) == 6
    assert backend.power(2, 3) == 8
    assert jnp.allclose(backend.exp(1), jnp.exp(1))
    assert jnp.allclose(backend.sqrt(4), 2)
    assert jnp.allclose(backend.gauss_pdf(0, 0, 1), 1.0 / math.sqrt(2 * math.pi))  # Standard normal PDF at 0
    assert jnp.allclose(backend.uniform_pdf(0.5, 0, 1), 1.0)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_mock_backend_operations_edge_cases(backend):
    # Test edge cases for backend operations
    assert backend.multiply(0, 3) == 0  # Multiplication with zero
    assert backend.power(2, 0) == 1  # Any number to the power of zero
    assert backend.power(0, 2) == 0  # Zero to any power
    assert jnp.allclose(backend.sqrt(0), 0)  # Square root of zero
    assert jnp.allclose(backend.gauss_pdf(0, 0, 0), 1.0)  # Simplified PDF
    assert jnp.allclose(backend.uniform_pdf(0.5, 0.5, 0.5), 1.0)  # Simplified uniform PDF