"""Tests for the factor module."""

import numpy as np
import pytest

from pyForwardFolding.factor import (
    AbstractFactor,
    DeltaGamma,
    FluxNorm,
    PowerLawFlux,
    SnowstormGauss,
    get_parameter_values,
    get_required_variable_values,
)


class MockFactor(AbstractFactor):
    """Mock factor for testing abstract functionality."""
    
    def __init__(self, name: str, param_mapping=None):
        super().__init__(name, param_mapping)
        self.factor_parameters = ["param1", "param2"]
        self.req_vars = ["var1", "var2"]
    
    def evaluate(self, input_variables, parameters):
        return np.ones(len(input_variables["var1"]))


class TestAbstractFactor:
    """Test the abstract factor base class."""
    
    def test_factor_initialization(self):
        """Test factor initialization."""
        factor = MockFactor("test_factor")
        assert factor.name == "test_factor"
        assert factor.required_variables == ["var1", "var2"]
        assert factor.factor_parameters == ["param1", "param2"]
    
    def test_parameter_mapping_default(self):
        """Test default parameter mapping."""
        factor = MockFactor("test_factor")
        expected_mapping = {"param1": "param1", "param2": "param2"}
        assert factor.parameter_mapping == expected_mapping
        assert factor.exposed_parameters == ["param1", "param2"]
    
    def test_parameter_mapping_custom(self):
        """Test custom parameter mapping."""
        mapping = {"param1": "custom_param1", "param2": "custom_param2"}
        factor = MockFactor("test_factor", param_mapping=mapping)
        assert factor.parameter_mapping == mapping
        assert factor.exposed_parameters == ["custom_param1", "custom_param2"]
    
    def test_construct_from_not_implemented(self):
        """Test that construct_from raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            AbstractFactor.construct_from({})


class TestUtilityFunctions:
    """Test utility functions for factors."""
    
    def test_get_required_variable_values(self):
        """Test getting required variable values."""
        factor = MockFactor("test_factor")
        input_variables = {
            "var1": np.array([1, 2, 3]),
            "var2": np.array([4, 5, 6]),
            "extra_var": np.array([7, 8, 9])
        }
        
        result = get_required_variable_values(factor, input_variables)
        expected = {
            "var1": np.array([1, 2, 3]),
            "var2": np.array([4, 5, 6])
        }
        
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])
    
    def test_get_parameter_values(self):
        """Test getting parameter values."""
        factor = MockFactor("test_factor")
        parameter_dict = {
            "param1": 1.0,
            "param2": 2.0,
            "extra_param": 3.0
        }
        
        result = get_parameter_values(factor, parameter_dict)
        expected = {"param1": 1.0, "param2": 2.0}
        assert result == expected
    
    def test_get_parameter_values_with_mapping(self):
        """Test getting parameter values with custom mapping."""
        mapping = {"param1": "custom_param1", "param2": "custom_param2"}
        factor = MockFactor("test_factor", param_mapping=mapping)
        parameter_dict = {
            "custom_param1": 1.0,
            "custom_param2": 2.0,
            "extra_param": 3.0
        }
        
        result = get_parameter_values(factor, parameter_dict)
        expected = {"param1": 1.0, "param2": 2.0}
        assert result == expected


class TestPowerLawFlux:
    """Test the PowerLawFlux factor."""
    
    def test_initialization(self):
        """Test PowerLawFlux initialization."""
        factor = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18
        )
        
        assert factor.name == "powerlaw"
        assert factor.pivot_energy == 1e5
        assert factor.baseline_norm == 1e-18
        assert "flux_norm" in factor.factor_parameters
        assert "spectral_index" in factor.factor_parameters
        assert "log10_true_energy" in factor.required_variables
    
    def test_evaluate_basic(self):
        """Test basic PowerLawFlux evaluation."""
        factor = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18
        )
        
        input_variables = {
            "log10_true_energy": np.array([4.0, 5.0, 6.0])  # log10(E) = 4, 5, 6
        }
        parameters = {
            "flux_norm": 1.0,
            "spectral_index": -2.0
        }
        
        result = factor.evaluate(input_variables, parameters)
        
        # Check that result has correct shape
        assert result.shape == (3,)
        # Check that result is positive
        assert np.all(result > 0)
        # Check that flux decreases with energy for negative spectral index
        assert result[0] > result[1] > result[2]
    
    def test_evaluate_with_param_mapping(self):
        """Test PowerLawFlux with parameter mapping."""
        param_mapping = {
            "flux_norm": "astro_norm",
            "spectral_index": "astro_index"
        }
        factor = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18,
            param_mapping=param_mapping
        )
        
        input_variables = {
            "log10_true_energy": np.array([5.0])
        }
        parameters = {
            "astro_norm": 2.0,
            "astro_index": -2.5
        }
        
        result = factor.evaluate(input_variables, parameters)
        assert result.shape == (1,)
        assert result[0] > 0


class TestFluxNorm:
    """Test the FluxNorm factor."""
    
    def test_initialization(self):
        """Test FluxNorm initialization."""
        factor = FluxNorm(name="norm_factor")
        
        assert factor.name == "norm_factor"
        assert "flux_norm" in factor.factor_parameters
        assert factor.required_variables == []
    
    def test_evaluate(self):
        """Test FluxNorm evaluation."""
        factor = FluxNorm(name="norm_factor")
        
        input_variables = {"dummy": np.array([1, 2, 3])}
        parameters = {"flux_norm": 2.5}
        
        result = factor.evaluate(input_variables, parameters)
        expected = np.array([2.5, 2.5, 2.5])
        np.testing.assert_array_almost_equal(result, expected)


class TestSnowstormGauss:
    """Test the SnowstormGauss factor."""
    
    def test_initialization(self):
        """Test SnowstormGauss initialization."""
        factor = SnowstormGauss(
            name="snowstorm",
            sys_gauss_width=0.1,
            sys_sim_bounds=[0.8, 1.2],
            req_variable_name="energy_scale"
        )
        
        assert factor.name == "snowstorm"
        assert factor.sys_gauss_width == 0.1
        assert factor.sys_sim_bounds == [0.8, 1.2]
        assert "energy_scale" in factor.required_variables
        assert "scale" in factor.factor_parameters
    
    def test_evaluate_no_effect(self):
        """Test SnowstormGauss with scale=1 (no effect)."""
        factor = SnowstormGauss(
            name="snowstorm",
            sys_gauss_width=0.1,
            sys_sim_bounds=[0.8, 1.2],
            req_variable_name="energy_scale"
        )
        
        input_variables = {
            "energy_scale": np.array([1.0, 1.0, 1.0])
        }
        parameters = {"scale": 1.0}
        
        result = factor.evaluate(input_variables, parameters)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_with_effect(self):
        """Test SnowstormGauss with scale != 1."""
        factor = SnowstormGauss(
            name="snowstorm",
            sys_gauss_width=0.1,
            sys_sim_bounds=[0.8, 1.2],
            req_variable_name="energy_scale"
        )
        
        input_variables = {
            "energy_scale": np.array([0.9, 1.0, 1.1])
        }
        parameters = {"scale": 1.1}
        
        result = factor.evaluate(input_variables, parameters)
        
        # Check that result has correct shape
        assert result.shape == (3,)
        # All results should be positive
        assert np.all(result > 0)


class TestDeltaGamma:
    """Test the DeltaGamma factor."""
    
    def test_initialization(self):
        """Test DeltaGamma initialization."""
        factor = DeltaGamma(
            name="delta_gamma",
            reference_energy=1e5
        )
        
        assert factor.name == "delta_gamma"
        assert factor.reference_energy == 1e5
        assert "log10_true_energy" in factor.required_variables
        assert "delta_gamma" in factor.factor_parameters
    
    def test_evaluate_no_change(self):
        """Test DeltaGamma with delta_gamma=0 (no change)."""
        factor = DeltaGamma(
            name="delta_gamma",
            reference_energy=1e5
        )
        
        input_variables = {
            "log10_true_energy": np.array([4.0, 5.0, 6.0])
        }
        parameters = {"delta_gamma": 0.0}
        
        result = factor.evaluate(input_variables, parameters)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_with_change(self):
        """Test DeltaGamma with non-zero delta_gamma."""
        factor = DeltaGamma(
            name="delta_gamma",
            reference_energy=1e5
        )
        
        input_variables = {
            "log10_true_energy": np.array([4.0, 5.0, 6.0])  # 10^4, 10^5, 10^6 GeV
        }
        parameters = {"delta_gamma": -0.1}
        
        result = factor.evaluate(input_variables, parameters)
        
        # Check that result has correct shape
        assert result.shape == (3,)
        # All results should be positive
        assert np.all(result > 0)
        # With negative delta_gamma, higher energies should have smaller factors
        assert result[0] > result[1] and result[1] > result[2]


class TestFactorConstruction:
    """Test factor construction from configuration."""
    
    def test_powerlaw_construct_from(self):
        """Test constructing PowerLawFlux from config."""
        config = {
            "name": "powerlaw",
            "type": "PowerLawFlux",
            "pivot_energy": 1e5,
            "baseline_norm": 1e-18,
            "param_mapping": {
                "flux_norm": "astro_norm",
                "spectral_index": "astro_index"
            }
        }
        
        factor = PowerLawFlux.construct_from(config)
        
        assert factor.name == "powerlaw"
        assert factor.pivot_energy == 1e5
        assert factor.baseline_norm == 1e-18
        assert factor.parameter_mapping["flux_norm"] == "astro_norm"
        assert factor.parameter_mapping["spectral_index"] == "astro_index"
    
    def test_flux_norm_construct_from(self):
        """Test constructing FluxNorm from config."""
        config = {
            "name": "norm_factor",
            "type": "FluxNorm",
            "param_mapping": {
                "flux_norm": "atmo_norm"
            }
        }
        
        factor = FluxNorm.construct_from(config)
        
        assert factor.name == "norm_factor"
        assert factor.parameter_mapping["flux_norm"] == "atmo_norm"
