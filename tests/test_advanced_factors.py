"""Enhanced tests for critical factor types."""

import numpy as np
import pytest

from pyForwardFolding.factor import (
    GradientReweight,
    ModelInterpolator,
    ScaledTemplate,
    SnowStormGradient,
    SoftCut,
    VetoThreshold,
)


class TestGradientReweight:
    """Test the GradientReweight factor."""
    
    def test_initialization(self):
        """Test GradientReweight initialization."""
        factor = GradientReweight(
            name="gradient_reweight",
            baseline_weight="base_weight",
            gradient_key_mapping={"param1": "grad_param1", "param2": "grad_param2"}
        )
        
        assert factor.name == "gradient_reweight"
        assert factor.baseline_weight == "base_weight"
        assert len(factor.factor_parameters) == 2
        assert "param1" in factor.factor_parameters
        assert "param2" in factor.factor_parameters
    
    def test_evaluate_no_gradient(self):
        """Test evaluation with zero gradients."""
        factor = GradientReweight(
            name="gradient_reweight",
            baseline_weight="base_weight",
            gradient_key_mapping={"param1": "grad_param1"}
        )
        
        input_variables = {
            "base_weight": np.array([1.0, 2.0, 3.0]),
            "grad_param1": np.array([0.0, 0.0, 0.0])  # No gradient
        }
        parameters = {"param1": 0.5}
        
        result = factor.evaluate(input_variables, parameters)
        expected = np.array([1.0, 1.0, 1.0])  # Should return 1 (no change)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_with_gradient(self):
        """Test evaluation with non-zero gradients."""
        factor = GradientReweight(
            name="gradient_reweight",
            baseline_weight="base_weight",
            gradient_key_mapping={"param1": "grad_param1"}
        )
        
        baseline = np.array([2.0, 4.0])
        gradient = np.array([0.1, -0.2])
        param_value = 1.0
        
        input_variables = {
            "base_weight": baseline,
            "grad_param1": gradient
        }
        parameters = {"param1": param_value}
        
        result = factor.evaluate(input_variables, parameters)
        
        # Expected: (baseline + param_value * gradient) / baseline
        expected = (baseline + param_value * gradient) / baseline
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_construct_from(self):
        """Test construction from configuration."""
        config = {
            "name": "test_gradient",
            "baseline_weight": "base_weight",
            "gradient_key_mapping": {"param1": "grad_param1", "param2": "grad_param2"}
        }
        
        factor = GradientReweight.construct_from(config)
        
        assert factor.name == "test_gradient"
        assert factor.baseline_weight == "base_weight"
        assert len(factor.factor_parameters) == 2


class TestModelInterpolator:
    """Test the ModelInterpolator factor."""
    
    def test_initialization(self):
        """Test ModelInterpolator initialization."""
        factor = ModelInterpolator(
            name="interpolator",
            baseline_weight="base_weight",
            alternative_weight="alt_weight"
        )
        
        assert factor.name == "interpolator"
        assert factor.base_key == "base_weight"
        assert factor.alt_key == "alt_weight"
        assert "lambda_int" in factor.factor_parameters
    
    def test_evaluate_no_interpolation(self):
        """Test evaluation with lambda_int = 0 (no interpolation)."""
        factor = ModelInterpolator(
            name="interpolator",
            baseline_weight="base_weight",
            alternative_weight="alt_weight"
        )
        
        input_variables = {
            "base_weight": np.array([1.0, 2.0, 3.0]),
            "alt_weight": np.array([2.0, 4.0, 6.0])
        }
        parameters = {"lambda_int": 0.0}
        
        result = factor.evaluate(input_variables, parameters)
        expected = np.array([1.0, 1.0, 1.0])  # Should return 1 (no change)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_full_interpolation(self):
        """Test evaluation with lambda_int = 1 (full interpolation)."""
        factor = ModelInterpolator(
            name="interpolator",
            baseline_weight="base_weight", 
            alternative_weight="alt_weight"
        )
        
        input_variables = {
            "base_weight": np.array([1.0, 2.0]),
            "alt_weight": np.array([2.0, 4.0])
        }
        parameters = {"lambda_int": 1.0}
        
        result = factor.evaluate(input_variables, parameters)
        expected = np.array([2.0, 2.0])  # alt_weight / base_weight
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_zero_baseline(self):
        """Test evaluation with zero baseline weight."""
        factor = ModelInterpolator(
            name="interpolator",
            baseline_weight="base_weight",
            alternative_weight="alt_weight"
        )
        
        input_variables = {
            "base_weight": np.array([0.0, 2.0]),
            "alt_weight": np.array([1.0, 4.0])
        }
        parameters = {"lambda_int": 0.5}
        
        result = factor.evaluate(input_variables, parameters)
        
        # First element should return 1 (zero baseline case)
        assert result[0] == 1.0
        # Second element should be normal interpolation
        assert result[1] != 1.0
    
    def test_construct_from(self):
        """Test construction from configuration."""
        config = {
            "name": "test_interpolator",
            "baseline_weight": "base",
            "alternative_weight": "alt"
        }
        
        factor = ModelInterpolator.construct_from(config)
        
        assert factor.name == "test_interpolator"
        assert factor.base_key == "base"
        assert factor.alt_key == "alt"


class TestSoftCut:
    """Test the SoftCut factor."""
    
    def test_initialization(self):
        """Test SoftCut initialization."""
        factor = SoftCut(
            name="soft_cut",
            cut_variable="energy",
            slope=10.0
        )
        
        assert factor.name == "soft_cut"
        assert factor.cut_variable == "energy"
        assert factor.slope == 10.0
        assert "soft_cut" in factor.factor_parameters
    
    def test_evaluate_sigmoid_behavior(self):
        """Test that evaluation produces sigmoid behavior."""
        factor = SoftCut(
            name="soft_cut",
            cut_variable="energy",
            slope=1.0
        )
        
        # Test points around the cut value
        input_variables = {
            "energy": np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        }
        parameters = {"soft_cut": 0.0}  # Cut at energy = 0
        
        result = factor.evaluate(input_variables, parameters)
        
        # Should be sigmoid curve - increasing with energy
        assert result[0] < result[1] < result[2] < result[3] < result[4]
        # Should be between 0 and 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        # Should be approximately 0.5 at cut value
        np.testing.assert_almost_equal(result[2], 0.5, decimal=3)
    
    def test_steep_slope(self):
        """Test behavior with steep slope (approaches step function)."""
        factor = SoftCut(
            name="soft_cut",
            cut_variable="energy",
            slope=100.0
        )
        
        input_variables = {
            "energy": np.array([0.9, 1.0, 1.1])
        }
        parameters = {"soft_cut": 1.0}
        
        result = factor.evaluate(input_variables, parameters)
        
        # With steep slope, should be nearly step-like
        assert result[0] < 0.1  # Well below cut
        assert result[2] > 0.9  # Well above cut
    
    def test_construct_from(self):
        """Test construction from configuration."""
        config = {
            "name": "test_soft_cut",
            "cut_variable": "zenith",
            "slope": 5.0
        }
        
        factor = SoftCut.construct_from(config)
        
        assert factor.name == "test_soft_cut"
        assert factor.cut_variable == "zenith"
        assert factor.slope == 5.0


class TestVetoThreshold:
    """Test the VetoThreshold factor."""
    
    def test_initialization(self):
        """Test VetoThreshold initialization."""
        factor = VetoThreshold(
            name="veto",
            threshold_a=1.0,
            threshold_b=2.0,
            threshold_c=0.5,
            rescale_energy=1e3,
            anchor_energy=1e5
        )
        
        assert factor.name == "veto"
        assert factor.a == 1.0
        assert factor.b == 2.0
        assert factor.c == 0.5
        assert factor.rescale_energy == 1e3
        assert factor.anchor_energy == 1e5
    
    def test_evaluate_basic(self):
        """Test basic evaluation functionality."""
        factor = VetoThreshold(
            name="veto",
            threshold_a=0.0,  # Simplified case
            threshold_b=0.0,
            threshold_c=0.0,
            rescale_energy=1e3,
            anchor_energy=1e5
        )
        
        input_variables = {}  # VetoThreshold doesn't need input variables
        parameters = {"e_threshold": 0.0}
        
        result = factor.evaluate(input_variables, parameters)
        
        # With simplified parameters, should return reasonable values
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)  # Should be positive reweighting factors
    
    def test_construct_from(self):
        """Test construction from configuration."""
        config = {
            "name": "test_veto",
            "threshold_a": 1.0,
            "threshold_b": 2.0,
            "threshold_c": 0.5,
            "rescale_energy": 1e3,
            "anchor_energy": 1e5
        }
        
        factor = VetoThreshold.construct_from(config)
        
        assert factor.name == "test_veto"
        assert factor.a == 1.0
        assert factor.threshold_b == 2.0
        assert factor.threshold_c == 0.5


class TestScaledTemplate:
    """Test the ScaledTemplate binned factor."""
    
    def test_initialization(self):
        """Test ScaledTemplate initialization."""
        template = np.array([1.0, 2.0, 3.0, 4.0])
        
        factor = ScaledTemplate(
            name="scaled_template",
            template=template
        )
        
        assert factor.name == "scaled_template"
        np.testing.assert_array_equal(factor.template, template)
        assert "template_norm" in factor.factor_parameters
    
    def test_evaluate_binned_basic(self):
        """Test basic binned evaluation."""
        template = np.array([1.0, 2.0, 3.0])
        
        factor = ScaledTemplate(
            name="scaled_template",
            template=template
        )
        
        binned_data = np.array([10.0, 20.0, 30.0])
        parameters = {"template_norm": 0.5}
        
        result = factor.evaluate_binned(binned_data, parameters)
        
        # Should add scaled template to binned data
        expected = binned_data + 0.5 * template
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_binned_zero_norm(self):
        """Test evaluation with zero normalization."""
        template = np.array([1.0, 2.0, 3.0])
        
        factor = ScaledTemplate(
            name="scaled_template",
            template=template
        )
        
        binned_data = np.array([10.0, 20.0, 30.0])
        parameters = {"template_norm": 0.0}
        
        result = factor.evaluate_binned(binned_data, parameters)
        
        # Should return original data unchanged
        np.testing.assert_array_almost_equal(result, binned_data)
    
    def test_construct_from(self):
        """Test construction from configuration."""
        template = [1.0, 2.0, 3.0]
        config = {
            "name": "test_template",
            "template": template
        }
        
        factor = ScaledTemplate.construct_from(config)
        
        assert factor.name == "test_template"
        np.testing.assert_array_equal(factor.template, np.array(template))


class TestSnowStormGradient:
    """Test the SnowStormGradient binned factor."""
    
    def test_initialization(self):
        """Test SnowStormGradient initialization."""
        factor = SnowStormGradient(
            name="snowstorm_gradient",
            gradient_template=np.array([0.1, 0.2, 0.3])
        )
        
        assert factor.name == "snowstorm_gradient"
        np.testing.assert_array_equal(factor.gradient_template, np.array([0.1, 0.2, 0.3]))
        assert "snowstorm_gradient" in factor.factor_parameters
    
    def test_evaluate_binned_basic(self):
        """Test basic binned evaluation."""
        gradient_template = np.array([0.1, -0.2, 0.3])
        
        factor = SnowStormGradient(
            name="snowstorm_gradient",
            gradient_template=gradient_template
        )
        
        binned_data = np.array([10.0, 20.0, 30.0])
        parameters = {"snowstorm_gradient": 2.0}
        
        result = factor.evaluate_binned(binned_data, parameters)
        
        # Should add parameter * gradient to data
        expected = binned_data + 2.0 * gradient_template
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_binned_zero_parameter(self):
        """Test evaluation with zero parameter."""
        gradient_template = np.array([0.1, 0.2, 0.3])
        
        factor = SnowStormGradient(
            name="snowstorm_gradient",
            gradient_template=gradient_template
        )
        
        binned_data = np.array([10.0, 20.0, 30.0])
        parameters = {"snowstorm_gradient": 0.0}
        
        result = factor.evaluate_binned(binned_data, parameters)
        
        # Should return original data unchanged
        np.testing.assert_array_almost_equal(result, binned_data)
    
    def test_construct_from(self):
        """Test construction from configuration."""
        gradient_template = [0.1, 0.2, 0.3]
        config = {
            "name": "test_snowstorm",
            "gradient_template": gradient_template
        }
        
        factor = SnowStormGradient.construct_from(config)
        
        assert factor.name == "test_snowstorm"
        np.testing.assert_array_equal(factor.gradient_template, np.array(gradient_template))


class TestFactorIntegration:
    """Integration tests for advanced factors."""
    
    def test_factor_chaining(self):
        """Test that multiple factors can be chained together."""
        # Create a soft cut factor
        soft_cut = SoftCut(
            name="energy_cut",
            cut_variable="energy", 
            slope=5.0
        )
        
        # Create an interpolator
        interpolator = ModelInterpolator(
            name="model_interp",
            baseline_weight="base_weight",
            alternative_weight="alt_weight"
        )
        
        # Sample data
        input_variables = {
            "energy": np.array([0.5, 1.5, 2.5]),
            "base_weight": np.array([1.0, 2.0, 3.0]),
            "alt_weight": np.array([1.5, 3.0, 4.5])
        }
        
        cut_params = {"soft_cut": 1.0}
        interp_params = {"lambda_int": 0.5}
        
        # Apply factors sequentially
        cut_result = soft_cut.evaluate(input_variables, cut_params)
        interp_result = interpolator.evaluate(input_variables, interp_params)
        
        # Both should produce valid results
        assert np.all(np.isfinite(cut_result))
        assert np.all(np.isfinite(interp_result))
        assert np.all(cut_result >= 0)
        assert np.all(cut_result <= 1)
        assert np.all(interp_result > 0)
    
    def test_parameter_mapping_consistency(self):
        """Test that parameter mapping works consistently across factors."""
        # Test with custom parameter mapping
        param_mapping = {"lambda_int": "custom_lambda"}
        
        interpolator = ModelInterpolator(
            name="mapped_interp",
            baseline_weight="base_weight",
            alternative_weight="alt_weight",
            param_mapping=param_mapping
        )
        
        assert interpolator.parameter_mapping == param_mapping
        assert interpolator.exposed_parameters == ["custom_lambda"]
    
    @pytest.mark.parametrize("factor_class,config", [
        (SoftCut, {"name": "test", "cut_variable": "energy", "slope": 1.0}),
        (VetoThreshold, {
            "name": "test", "threshold_a": 1.0, "threshold_b": 2.0, 
            "threshold_c": 0.5, "rescale_energy": 1e3, "anchor_energy": 1e5
        }),
        (ScaledTemplate, {"name": "test", "template": [1.0, 2.0, 3.0]}),
        (SnowStormGradient, {"name": "test", "gradient_template": [0.1, 0.2, 0.3]}),
    ])
    def test_factor_construct_from_pattern(self, factor_class, config):
        """Test that all factors follow the construct_from pattern consistently."""
        factor = factor_class.construct_from(config)
        
        assert factor.name == "test"
        assert hasattr(factor, "factor_parameters")
        assert hasattr(factor, "required_variables")
        assert hasattr(factor, "exposed_parameters")
