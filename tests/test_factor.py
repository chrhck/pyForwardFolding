import pytest
import jax.numpy as jnp
from pyForwardFolding.factor import (
    EnergyScaler,
    PowerLawFlux,
    FluxNorm,
    SnowstormGauss,
    DeltaGamma,
    get_required_variable_values,
    get_exposed_variable_values,
)
from pyForwardFolding.backend import JAXBackend, Backend

# Example: Add a mock backend for testing purposes
class MockBackend(Backend):
    def array(self, data):
        return data

    def multiply(self, a, b):
        return a * b

    def power(self, a, b):
        return a ** b

    def exp(self, a):
        return exp(a)

    def sqrt(self, a):
        return sqrt(a)

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
def test_power_law_flux(backend):
    factor = PowerLawFlux(name="power_law", pivot_energy=1.0, baseline_norm=1.0)
    input_variables = {"true_energy": 10.0}
    exposed_variables = {"power_law": {"flux_norm": 2.0, "spectral_index": 1.0}}
    output = backend.array([1.0, 1.0, 1.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([0.2, 0.2, 0.2])  # flux_norm * (true_energy / pivot_energy)^(-spectral_index)
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
def test_delta_gamma(backend):
    factor = DeltaGamma(name="delta_gamma", reference_energy=1.0)
    input_variables = {"true_energy": 10.0}
    exposed_variables = {"delta_gamma": {"delta_gamma": 1.0}}
    output = backend.array([1.0, 1.0, 1.0])

    result = factor.evaluate(output, input_variables, exposed_variables)
    expected = backend.array([0.1, 0.1, 0.1])  # (true_energy / reference_energy)^(-delta_gamma)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("backend", [JAXBackend(), MockBackend()])
def test_backend_operations(backend):
    # Test backend operations to ensure correctness
    assert backend.multiply(2, 3) == 6
    assert backend.power(2, 3) == 8
    assert jnp.allclose(backend.exp(1), jnp.exp(1))
    assert jnp.allclose(backend.sqrt(4), 2)
    assert jnp.allclose(backend.gauss_pdf(0, 0, 1), 1.0 / sqrt(2 * pi))  # Standard normal PDF at 0
    assert jnp.allclose(backend.uniform_pdf(0.5, 0, 1), 1.0)