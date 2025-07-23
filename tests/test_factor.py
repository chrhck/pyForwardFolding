"""
Tests for pyForwardFolding.factor module.

This module contains comprehensive tests for all factor classes and utility functions
in the factor module, including:
- AbstractFactor base class
- All concrete factor implementations
- Utility functions for parameter and variable handling
- Factor construction from configuration
"""

import pickle
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest

from pyForwardFolding.backend import backend
from pyForwardFolding.binning import AbstractBinning
from pyForwardFolding.factor import (
    AbstractBinnedFactor,
    AbstractFactor,
    AbstractUnbinnedFactor,
    DeltaGamma,
    FluxNorm,
    GradientReweight,
    ModelInterpolator,
    PowerLawFlux,
    ScaledTemplate,
    SnowstormGauss,
    SnowStormGradient,
    SoftCut,
    VetoThreshold,
    get_parameter_values,
    get_required_variable_values,
)


class TestAbstractFactor:
    """Test the AbstractFactor base class."""

    def test_init_basic(self):
        """Test basic initialization of AbstractFactor."""
        factor = AbstractFactor("test_factor")
        assert factor.name == "test_factor"
        assert factor.param_mapping is None
        assert factor.factor_parameters == []

    def test_init_with_param_mapping(self):
        """Test initialization with parameter mapping."""
        param_mapping = {"internal_param": "external_param"}
        factor = AbstractFactor("test_factor", param_mapping)
        assert factor.name == "test_factor"
        assert factor.param_mapping == param_mapping

    def test_parameter_mapping_property_none(self):
        """Test parameter_mapping property when param_mapping is None."""
        factor = AbstractFactor("test_factor")
        factor.factor_parameters = ["param1", "param2"]
        expected = {"param1": "param1", "param2": "param2"}
        assert factor.parameter_mapping == expected

    def test_parameter_mapping_property_with_mapping(self):
        """Test parameter_mapping property with explicit mapping."""
        param_mapping = {"internal_param": "external_param"}
        factor = AbstractFactor("test_factor", param_mapping)
        assert factor.parameter_mapping == param_mapping

    def test_exposed_parameters_property(self):
        """Test the exposed_parameters property."""
        param_mapping = {"internal1": "external1", "internal2": "external2"}
        factor = AbstractFactor("test_factor", param_mapping)
        assert factor.exposed_parameters == ["external1", "external2"]

    def test_evaluate_not_implemented(self):
        """Test that evaluate method raises NotImplementedError."""
        factor = AbstractFactor("test_factor")
        with pytest.raises(NotImplementedError):
            factor.evaluate({}, {})


class TestAbstractUnbinnedFactor:
    """Test the AbstractUnbinnedFactor base class."""

    def test_init_basic(self):
        """Test basic initialization of AbstractUnbinnedFactor."""
        factor = AbstractUnbinnedFactor("test_factor")
        assert factor.name == "test_factor"
        assert factor.param_mapping is None
        assert factor.factor_parameters == []
        assert factor.req_vars == []

    def test_init_with_param_mapping(self):
        """Test initialization with parameter mapping."""
        param_mapping = {"internal_param": "external_param"}
        factor = AbstractUnbinnedFactor("test_factor", param_mapping)
        assert factor.name == "test_factor"
        assert factor.param_mapping == param_mapping

    def test_required_variables_property(self):
        """Test the required_variables property."""
        factor = AbstractUnbinnedFactor("test_factor")
        factor.req_vars = ["var1", "var2"]
        assert factor.required_variables == ["var1", "var2"]

    def test_evaluate_not_implemented(self):
        """Test that evaluate method raises NotImplementedError."""
        factor = AbstractUnbinnedFactor("test_factor")
        with pytest.raises(NotImplementedError):
            factor.evaluate({}, {})

    def test_construct_from_missing_type(self):
        """Test construct_from with missing type key."""
        config = {"name": "test"}
        with pytest.raises(ValueError, match="Configuration must contain a 'type' key"):
            AbstractUnbinnedFactor.construct_from(config)

    def test_construct_from_unknown_type(self):
        """Test construct_from with unknown factor type."""
        config = {"type": "UnknownFactor", "name": "test"}
        with pytest.raises(ValueError, match="Unknown factor type: UnknownFactor"):
            AbstractUnbinnedFactor.construct_from(config)

    def test_construct_from_valid_type(self):
        """Test construct_from with valid factor type."""
        config = {
            "type": "FluxNorm",
            "name": "test_flux_norm"
        }
        factor = AbstractUnbinnedFactor.construct_from(config)
        assert isinstance(factor, FluxNorm)
        assert factor.name == "test_flux_norm"


class TestUtilityFunctions:
    """Test utility functions for factor operations."""

    def test_get_required_variable_values(self):
        """Test get_required_variable_values function."""
        factor = Mock()
        factor.required_variables = ["var1", "var2"]
        
        input_vars = {
            "var1": backend.array([1, 2, 3]),
            "var2": backend.array([4, 5, 6]),
            "var3": backend.array([7, 8, 9])  # Should not be included
        }
        
        result = get_required_variable_values(factor, input_vars)
        assert len(result) == 2
        assert "var1" in result
        assert "var2" in result
        assert "var3" not in result

    def test_get_parameter_values(self):
        """Test get_parameter_values function."""
        factor = Mock()
        factor.parameter_mapping = {"internal_param1": "external_param1", "internal_param2": "external_param2"}
        
        param_dict = {
            "external_param1": 1.5,
            "external_param2": 2.5,
            "unused_param": 3.5  # Should not be included
        }
        
        result = get_parameter_values(factor, param_dict)
        expected = {"internal_param1": 1.5, "internal_param2": 2.5}
        assert result == expected


class TestPowerLawFlux:
    """Test PowerLawFlux factor."""

    def test_init(self):
        """Test PowerLawFlux initialization."""
        factor = PowerLawFlux("power_law", pivot_energy=100.0, baseline_norm=1e-11)
        assert factor.name == "power_law"
        assert factor.pivot_energy == 100.0
        assert factor.baseline_norm == 1e-11
        assert factor.factor_parameters == ["flux_norm", "spectral_index"]
        assert factor.req_vars == ["true_energy"]

    def test_init_with_param_mapping(self):
        """Test PowerLawFlux initialization with parameter mapping."""
        param_mapping = {"flux_norm": "norm", "spectral_index": "gamma"}
        factor = PowerLawFlux("power_law", 100.0, 1e-11, param_mapping)
        assert factor.parameter_mapping == param_mapping

    def test_construct_from(self):
        """Test PowerLawFlux construction from config."""
        config = {
            "name": "test_power_law",
            "pivot_energy": 100.0,
            "baseline_norm": 1e-11
        }
        factor = PowerLawFlux.construct_from(config)
        assert factor.name == "test_power_law"
        assert factor.pivot_energy == 100.0
        assert factor.baseline_norm == 1e-11

    def test_construct_from_with_param_mapping(self):
        """Test PowerLawFlux construction from config with parameter mapping."""
        config = {
            "name": "test_power_law",
            "pivot_energy": 100.0,
            "baseline_norm": 1e-11,
            "param_mapping": {"flux_norm": "norm", "spectral_index": "gamma"}
        }
        factor = PowerLawFlux.construct_from(config)
        assert factor.parameter_mapping == {"flux_norm": "norm", "spectral_index": "gamma"}

    def test_evaluate(self):
        """Test PowerLawFlux evaluation."""
        factor = PowerLawFlux("power_law", pivot_energy=100.0, baseline_norm=1e-11)
        
        input_variables = {
            "true_energy": backend.array([50.0, 100.0, 200.0])
        }
        parameters = {
            "flux_norm": 1.0,
            "spectral_index": 2.0
        }
        
        result = factor.evaluate(input_variables, parameters)
        
        # Expected: flux_norm * baseline_norm * (energy/pivot)^(-spectral_index)
        expected_energy_ratio = backend.array([50.0/100.0, 100.0/100.0, 200.0/100.0])
        expected = 1.0 * 1e-11 * backend.power(expected_energy_ratio, -2.0)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestFluxNorm:
    """Test FluxNorm factor."""

    def test_init(self):
        """Test FluxNorm initialization."""
        factor = FluxNorm("flux_norm")
        assert factor.name == "flux_norm"
        assert factor.factor_parameters == ["flux_norm"]
        assert factor.req_vars == []

    def test_construct_from(self):
        """Test FluxNorm construction from config."""
        config = {"name": "test_flux_norm"}
        factor = FluxNorm.construct_from(config)
        assert factor.name == "test_flux_norm"

    def test_evaluate(self):
        """Test FluxNorm evaluation."""
        factor = FluxNorm("flux_norm")
        
        input_variables = {}
        parameters = {"flux_norm": 2.5}
        
        result = factor.evaluate(input_variables, parameters)
        expected = backend.array(2.5)
        
        np.testing.assert_array_equal(result, expected)


class TestSnowstormGauss:
    """Test SnowstormGauss factor."""

    def test_init(self):
        """Test SnowstormGauss initialization."""
        factor = SnowstormGauss(
            name="snowstorm",
            sys_gauss_width=0.1,
            sys_sim_bounds=(-1.0, 1.0),
            req_variable_name="sys_var"
        )
        assert factor.name == "snowstorm"
        assert factor.sys_gauss_width == 0.1
        assert factor.sys_sim_bounds == (-1.0, 1.0)
        assert factor.req_vars == ["sys_var"]
        assert factor.factor_parameters == ["scale"]

    def test_construct_from(self):
        """Test SnowstormGauss construction from config."""
        config = {
            "name": "test_snowstorm",
            "sys_gauss_width": 0.1,
            "sys_sim_bounds": [-1.0, 1.0],
            "req_variable_name": "sys_var"
        }
        factor = SnowstormGauss.construct_from(config)
        assert factor.name == "test_snowstorm"
        assert factor.sys_gauss_width == 0.1
        assert factor.sys_sim_bounds == (-1.0, 1.0)
        assert factor.req_vars == ["sys_var"]

    def test_evaluate(self):
        """Test SnowstormGauss evaluation."""
        factor = SnowstormGauss(
            name="snowstorm",
            sys_gauss_width=0.1,
            sys_sim_bounds=(-1.0, 1.0),
            req_variable_name="sys_var"
        )
        
        input_variables = {
            "sys_var": backend.array([0.0, 0.5, -0.5])
        }
        parameters = {"scale": 0.2}
        
        result = factor.evaluate(input_variables, parameters)
        
        # Test that result has the correct shape
        assert result.shape == (3,)
        # Test that all values are positive (as expected for a probability ratio)
        assert np.all(result > 0)


class TestDeltaGamma:
    """Test DeltaGamma factor."""

    def test_init(self):
        """Test DeltaGamma initialization."""
        factor = DeltaGamma("delta_gamma", reference_energy=1.0)
        assert factor.name == "delta_gamma"
        assert factor.reference_energy == 1.0
        assert factor.factor_parameters == ["delta_gamma"]
        assert factor.req_vars == ["true_energy", "median_energy"]

    def test_construct_from(self):
        """Test DeltaGamma construction from config."""
        config = {
            "name": "test_delta_gamma",
            "reference_energy": 100.0
        }
        factor = DeltaGamma.construct_from(config)
        assert factor.name == "test_delta_gamma"
        assert factor.reference_energy == 100.0

    def test_evaluate(self):
        """Test DeltaGamma evaluation."""
        factor = DeltaGamma("delta_gamma", reference_energy=100.0)
        
        input_variables = {
            "true_energy": backend.array([50.0, 100.0, 200.0]),
            "median_energy": backend.array([60.0, 110.0, 190.0])  # Not used in current implementation
        }
        parameters = {"delta_gamma": 0.1}
        
        result = factor.evaluate(input_variables, parameters)
        
        # Expected: (true_energy / reference_energy)^(-delta_gamma)
        expected_energy_ratio = backend.array([50.0/100.0, 100.0/100.0, 200.0/100.0])
        expected = backend.power(expected_energy_ratio, -0.1)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestModelInterpolator:
    """Test ModelInterpolator factor."""

    def test_init(self):
        """Test ModelInterpolator initialization."""
        factor = ModelInterpolator("interpolator", "baseline_weight", "alternative_weight")
        assert factor.name == "interpolator"
        assert factor.base_key == "baseline_weight"
        assert factor.alt_key == "alternative_weight"
        assert factor.req_vars == ["baseline_weight", "alternative_weight"]
        assert factor.factor_parameters == ["lambda_int"]

    def test_construct_from(self):
        """Test ModelInterpolator construction from config."""
        config = {
            "name": "test_interpolator",
            "baseline_weight": "base_w",
            "alternative_weight": "alt_w"
        }
        factor = ModelInterpolator.construct_from(config)
        assert factor.name == "test_interpolator"
        assert factor.base_key == "base_w"
        assert factor.alt_key == "alt_w"

    def test_evaluate_normal_case(self):
        """Test ModelInterpolator evaluation for normal case."""
        factor = ModelInterpolator("interpolator", "baseline_weight", "alternative_weight")
        
        input_variables = {
            "baseline_weight": backend.array([1.0, 2.0, 3.0]),
            "alternative_weight": backend.array([1.5, 2.5, 3.5])
        }
        parameters = {"lambda_int": 0.5}
        
        result = factor.evaluate(input_variables, parameters)
        
        # Test that result has the correct shape
        assert result.shape == (3,)
        # Test that interpolation works as expected (values between baseline adjustments)
        assert np.all(result > 0)

    def test_evaluate_zero_baseline(self):
        """Test ModelInterpolator evaluation when baseline weight is zero."""
        factor = ModelInterpolator("interpolator", "baseline_weight", "alternative_weight")
        
        input_variables = {
            "baseline_weight": backend.array([0.0, 2.0, 0.0]),
            "alternative_weight": backend.array([1.5, 2.5, 3.5])
        }
        parameters = {"lambda_int": 0.5}
        
        result = factor.evaluate(input_variables, parameters)
        
        # When baseline weight is 0, result should be 1
        np.testing.assert_array_equal(result[0], 1.0)
        np.testing.assert_array_equal(result[2], 1.0)


class TestGradientReweight:
    """Test GradientReweight factor."""

    def test_init(self):
        """Test GradientReweight initialization."""
        gradient_mapping = {"param1": "grad1", "param2": "grad2"}
        factor = GradientReweight("grad_reweight", gradient_mapping, "baseline")
        assert factor.name == "grad_reweight"
        assert factor.baseline_weight == "baseline"
        assert factor.grad_key_map == gradient_mapping
        assert factor.req_vars == ["grad1", "grad2", "baseline"]
        assert factor.factor_parameters == ["param1", "param2"]

    def test_construct_from(self):
        """Test GradientReweight construction from config."""
        config = {
            "name": "test_grad_reweight",
            "gradient_key_mapping": {"param1": "grad1", "param2": "grad2"},
            "baseline_weight": "baseline"
        }
        factor = GradientReweight.construct_from(config)
        assert factor.name == "test_grad_reweight"
        assert factor.grad_key_map == {"param1": "grad1", "param2": "grad2"}
        assert factor.baseline_weight == "baseline"

    def test_evaluate(self):
        """Test GradientReweight evaluation."""
        gradient_mapping = {"param1": "grad1", "param2": "grad2"}
        factor = GradientReweight("grad_reweight", gradient_mapping, "baseline")
        
        input_variables = {
            "grad1": backend.array([0.1, 0.2, 0.3]),
            "grad2": backend.array([0.05, 0.1, 0.15]),
            "baseline": backend.array([1.0, 2.0, 3.0])
        }
        parameters = {"param1": 0.5, "param2": 0.2}
        
        result = factor.evaluate(input_variables, parameters)
        
        # Expected: (baseline + param1*grad1 + param2*grad2) / baseline
        expected_numerator = input_variables["baseline"] + \
                           0.5 * input_variables["grad1"] + \
                           0.2 * input_variables["grad2"]
        expected = expected_numerator / input_variables["baseline"]
        
        np.testing.assert_array_almost_equal(result, expected)


class TestVetoThreshold:
    """Test VetoThreshold factor."""

    def test_init(self):
        """Test VetoThreshold initialization."""
        factor = VetoThreshold(
            name="veto_threshold",
            threshold_a="coeff_a",
            threshold_b="coeff_b", 
            threshold_c="coeff_c",
            rescale_energy=100.0,
            anchor_energy=50.0
        )
        assert factor.name == "veto_threshold"
        assert factor.a == "coeff_a"
        assert factor.b == "coeff_b"
        assert factor.c == "coeff_c"
        assert factor.e_rescale == 100.0
        assert factor.e_anchor == 50.0
        assert factor.req_vars == ["coeff_a", "coeff_b", "coeff_c"]
        assert factor.factor_parameters == ["e_threshold"]

    def test_construct_from(self):
        """Test VetoThreshold construction from config."""
        config = {
            "name": "test_veto_threshold",
            "threshold_a": "coeff_a",
            "threshold_b": "coeff_b",
            "threshold_c": "coeff_c",
            "rescale_energy": 100.0,
            "anchor_energy": 50.0
        }
        factor = VetoThreshold.construct_from(config)
        assert factor.name == "test_veto_threshold"
        assert factor.a == "coeff_a"
        assert factor.e_rescale == 100.0
        assert factor.e_anchor == 50.0

    def test_evaluate(self):
        """Test VetoThreshold evaluation."""
        factor = VetoThreshold(
            name="veto_threshold",
            threshold_a="coeff_a",
            threshold_b="coeff_b",
            threshold_c="coeff_c", 
            rescale_energy=100.0,
            anchor_energy=50.0
        )
        
        input_variables = {
            "coeff_a": backend.array([1.0, 1.1, 0.9]),
            "coeff_b": backend.array([0.1, 0.15, 0.05]),
            "coeff_c": backend.array([0.01, 0.02, 0.005])
        }
        parameters = {"e_threshold": 0.0}  # log10(threshold energy / 100 GeV)
        
        result = factor.evaluate(input_variables, parameters)
        
        # Test that result has the correct shape
        assert result.shape == (3,)
        # Test that all values are positive (as expected for a reweight factor)
        assert np.all(result > 0)


class TestSoftCut:
    """Test SoftCut factor."""

    def test_init(self):
        """Test SoftCut initialization."""
        factor = SoftCut("soft_cut", "cut_variable", slope=10.0)
        assert factor.name == "soft_cut"
        assert factor.cut_variable == "cut_variable"
        assert factor.slope == 10.0
        assert factor.factor_parameters == ["soft_cut"]
        assert factor.req_vars == ["cut_variable"]

    def test_construct_from(self):
        """Test SoftCut construction from config."""
        config = {
            "name": "test_soft_cut",
            "cut_variable": "energy",
            "slope": 5.0
        }
        factor = SoftCut.construct_from(config)
        assert factor.name == "test_soft_cut"
        assert factor.cut_variable == "energy"
        assert factor.slope == 5.0

    def test_evaluate(self):
        """Test SoftCut evaluation."""
        factor = SoftCut("soft_cut", "energy", slope=10.0)
        
        input_variables = {
            "energy": backend.array([1.0, 5.0, 10.0])
        }
        parameters = {"soft_cut": 5.0}
        
        result = factor.evaluate(input_variables, parameters)
        
        # Test that result has the correct shape
        assert result.shape == (3,)
        # Test that sigmoid produces values between 0 and 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        # Test that cut value at 5.0 gives approximately 0.5
        np.testing.assert_almost_equal(result[1], 0.5, decimal=5)


class TestAbstractBinnedFactor:
    """Test AbstractBinnedFactor base class."""

    def test_init(self):
        """Test AbstractBinnedFactor initialization."""
        mock_binning = Mock(spec=AbstractBinning)
        factor = AbstractBinnedFactor("binned_factor", mock_binning)
        assert factor.name == "binned_factor"
        assert factor.binning == mock_binning

    def test_construct_from_missing_type(self):
        """Test construct_from with missing type key."""
        mock_binning = Mock(spec=AbstractBinning)
        config = {"name": "test"}
        with pytest.raises(ValueError, match="Configuration must contain a 'type' key"):
            AbstractBinnedFactor.construct_from(config, mock_binning)

    def test_construct_from_unknown_type(self):
        """Test construct_from with unknown factor type."""
        mock_binning = Mock(spec=AbstractBinning)
        config = {"type": "UnknownBinnedFactor", "name": "test"}
        with pytest.raises(ValueError, match="Unknown factor type: UnknownBinnedFactor"):
            AbstractBinnedFactor.construct_from(config, mock_binning)


class TestSnowStormGradient:
    """Test SnowStormGradient factor."""

    def test_init_dimension_mismatch(self):
        """Test SnowStormGradient initialization with dimension mismatch."""
        mock_binning = Mock(spec=AbstractBinning)
        mock_binning.hist_dims = [10, 20]  # 2D binning
        
        # Create a temporary pickle file with mismatched dimensions
        gradient_data = {
            "binning": [np.linspace(0, 1, 11), np.linspace(0, 1, 16)],  # Different dimensions (10, 15)
            "livetime": 1000.0
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(gradient_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Mismatch between binning dimensions"):
                SnowStormGradient(
                    name="gradient_factor",
                    binning=mock_binning,
                    parameters=["param1"],
                    gradient_names=["grad1"],
                    default=[0.0],
                    split_values=[0.0],
                    gradient_pickle=temp_file
                )
        finally:
            import os
            os.unlink(temp_file)

    def test_construct_from(self):
        """Test SnowStormGradient construction from config."""
        mock_binning = Mock(spec=AbstractBinning)
        mock_binning.hist_dims = [10, 20]
        
        # Create a temporary pickle file with correct dimensions
        gradient_data = {
            "binning": [np.linspace(0, 1, 11), np.linspace(0, 1, 21)],  # Matching dimensions (10, 20)
            "livetime": 1000.0,
            "grad1": {
                "gradient": np.random.random((10, 20)),
                "gradient_error": np.random.random((10, 20))
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(gradient_data, f)
            temp_file = f.name
        
        try:
            config = {
                "name": "test_gradient",
                "parameters": ["param1"],
                "gradient_names": ["grad1"],
                "default": [0.0],
                "split_values": [0.0],
                "gradient_pickle": temp_file
            }
            
            factor = SnowStormGradient.construct_from(config, mock_binning)
            assert factor.name == "test_gradient"
            assert factor.factor_parameters == ["param1"]
            assert factor.gradient_names == ["grad1"]
        finally:
            import os
            os.unlink(temp_file)


class TestScaledTemplate:
    """Test ScaledTemplate factor."""

    def test_init_and_construct_from(self):
        """Test ScaledTemplate initialization and construction from config."""
        mock_binning = Mock(spec=AbstractBinning)
        mock_binning.hist_dims = [10, 20]
        
        # Create a temporary pickle file with template data
        template_data = {
            "template": np.random.random(200),  # Flattened 10x20 array
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(template_data, f)
            temp_file = f.name
        
        try:
            config = {
                "name": "test_template",
                "template_file": temp_file
            }
            
            factor = ScaledTemplate.construct_from(config, mock_binning)
            assert factor.name == "test_template"
            assert factor.factor_parameters == ["template_norm"]
            
            # Test evaluation
            input_variables = {}
            parameters = {"template_norm": 2.0}
            
            result, fluctuation = factor.evaluate(input_variables, parameters)
            
            # Test that result has the correct shape
            assert result.shape == (10, 20)
            assert fluctuation is None  # No template_fluctuation in test data
            
        finally:
            import os
            os.unlink(temp_file)

    def test_evaluate_with_fluctuation(self):
        """Test ScaledTemplate evaluation with template fluctuation."""
        mock_binning = Mock(spec=AbstractBinning)
        mock_binning.hist_dims = [5, 4]
        
        # Create template data with fluctuation
        template_data = {
            "template": np.random.random(20),  # Flattened 5x4 array
            "template_fluctuation": np.random.random(20)
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(template_data, f)
            temp_file = f.name
        
        try:
            factor = ScaledTemplate("template_factor", mock_binning, temp_file)
            
            input_variables = {}
            parameters = {"template_norm": 3.0}
            
            result, fluctuation = factor.evaluate(input_variables, parameters)
            
            # Test that result and fluctuation have the correct shapes
            assert result.shape == (5, 4)
            assert fluctuation is not None
            assert fluctuation.shape == (5, 4)
            
        finally:
            import os
            os.unlink(temp_file)


class TestFactorIntegration:
    """Integration tests for factor classes working together."""

    def test_multiple_factors_with_same_variables(self):
        """Test that multiple factors can share the same variables."""
        # Create two factors that use the same energy variable
        power_law = PowerLawFlux("power_law", 100.0, 1e-11)
        delta_gamma = DeltaGamma("delta_gamma", reference_energy=100.0)
        
        input_variables = {
            "true_energy": backend.array([50.0, 100.0, 200.0]),
            "median_energy": backend.array([60.0, 110.0, 190.0])
        }
        
        power_law_params = {"flux_norm": 1.0, "spectral_index": 2.0}
        delta_gamma_params = {"delta_gamma": 0.1}
        
        result1 = power_law.evaluate(input_variables, power_law_params)
        result2 = delta_gamma.evaluate(input_variables, delta_gamma_params)
        
        # Both should produce valid results
        assert result1.shape == (3,)
        assert result2.shape == (3,)
        assert np.all(result1 > 0)
        assert np.all(result2 > 0)

    def test_factor_chain_evaluation(self):
        """Test chaining factor evaluations (multiplying results)."""
        flux_norm = FluxNorm("flux_norm")
        power_law = PowerLawFlux("power_law", 100.0, 1e-11)
        
        input_variables = {
            "true_energy": backend.array([100.0])
        }
        
        flux_params = {"flux_norm": 2.0}
        power_params = {"flux_norm": 1.0, "spectral_index": 2.0}
        
        flux_result = flux_norm.evaluate({}, flux_params)
        power_result = power_law.evaluate(input_variables, power_params)
        
        # Combine results
        combined = flux_result * power_result
        
        # Should be 2.0 * (1.0 * 1e-11 * (100/100)^(-2.0)) = 2.0 * 1e-11
        expected = 2.0 * 1e-11
        np.testing.assert_almost_equal(combined, expected)

    def test_parameter_mapping_consistency(self):
        """Test that parameter mapping works consistently across factors."""
        param_mapping = {"flux_norm": "external_norm", "spectral_index": "external_gamma"}
        
        power_law = PowerLawFlux("power_law", 100.0, 1e-11, param_mapping)
        
        # Parameters should use the external names
        external_params = {"external_norm": 1.5, "external_gamma": 2.5}
        
        input_variables = {"true_energy": backend.array([100.0])}
        
        result = power_law.evaluate(input_variables, external_params)
        
        # Should use the mapped parameter values
        expected = 1.5 * 1e-11 * backend.power(1.0, -2.5)
        np.testing.assert_almost_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
