"""Tests for the model and model_component modules."""

import numpy as np
import pytest

from pyForwardFolding.factor import FluxNorm, PowerLawFlux
from pyForwardFolding.model import Model
from pyForwardFolding.model_component import ModelComponent


class TestModelComponent:
    """Test the ModelComponent class."""
    
    @pytest.fixture
    def sample_factors(self):
        """Provide sample factors for testing."""
        powerlaw = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18
        )
        norm = FluxNorm(name="norm")
        return [powerlaw, norm]
    
    def test_initialization(self, sample_factors):
        """Test ModelComponent initialization."""
        component = ModelComponent("astro", sample_factors)
        
        assert component.name == "astro"
        assert len(component.factors) == 2
        assert component.factors[0].name == "powerlaw"
        assert component.factors[1].name == "norm"
    
    def test_required_variables(self, sample_factors):
        """Test required variables property."""
        component = ModelComponent("astro", sample_factors)
        required_vars = component.required_variables
        
        # PowerLawFlux requires log10_true_energy, FluxNorm requires nothing
        assert "log10_true_energy" in required_vars
        assert len(required_vars) == 1
    
    def test_exposed_parameters(self, sample_factors):
        """Test exposed parameters property."""
        component = ModelComponent("astro", sample_factors)
        exposed_params = component.exposed_parameters
        
        # PowerLawFlux exposes flux_norm and spectral_index, FluxNorm exposes flux_norm
        expected_params = {"flux_norm", "spectral_index"}
        assert exposed_params == expected_params
    
    def test_parameter_mapping(self, sample_factors):
        """Test parameter mapping property."""
        component = ModelComponent("astro", sample_factors)
        param_mapping = component.parameter_mapping
        
        expected_mapping = {
            "powerlaw": {"flux_norm": "flux_norm", "spectral_index": "spectral_index"},
            "norm": {"flux_norm": "flux_norm"}
        }
        assert param_mapping == expected_mapping
    
    def test_evaluate(self, sample_factors):
        """Test component evaluation."""
        component = ModelComponent("astro", sample_factors)
        
        input_variables = {
            "log10_true_energy": np.array([4.0, 5.0, 6.0])
        }
        parameters = {
            "flux_norm": 2.0,
            "spectral_index": -2.0
        }
        
        result = component.evaluate(input_variables, parameters)
        
        # Result should be the product of all factor evaluations
        assert result.shape == (3,)
        assert np.all(result > 0)
        # Should be flux_norm * powerlaw_result * norm_result = 2.0 * powerlaw_result * 2.0
        assert np.all(result > 0)


class TestModel:
    """Test the Model class."""
    
    @pytest.fixture
    def sample_components(self):
        """Provide sample components for testing."""
        # Create factors
        powerlaw = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18
        )
        norm1 = FluxNorm(name="norm1")
        norm2 = FluxNorm(name="norm2")
        
        # Create components
        astro_component = ModelComponent("astro", [powerlaw, norm1])
        atmo_component = ModelComponent("atmo", [norm2])
        
        return [astro_component, atmo_component]
    
    def test_initialization(self, sample_components):
        """Test Model initialization."""
        baseline_weights = ["astro_weight", "atmo_weight"]
        model = Model("test_model", sample_components, baseline_weights)
        
        assert model.name == "test_model"
        assert len(model.components) == 2
        assert model.baseline_weights == baseline_weights
    
    def test_initialization_duplicate_names(self):
        """Test that duplicate component names raise an error."""
        factor = FluxNorm(name="norm")
        component1 = ModelComponent("same_name", [factor])
        component2 = ModelComponent("same_name", [factor])
        
        with pytest.raises(ValueError, match="Model components must have unique names"):
            Model("test_model", [component1, component2], ["weight1", "weight2"])
    
    def test_from_pairs_constructor(self, sample_components):
        """Test the from_pairs class method."""
        component_pairs = [
            ("astro_weight", sample_components[0]),
            ("atmo_weight", sample_components[1])
        ]
        
        model = Model.from_pairs("test_model", component_pairs)
        
        assert model.name == "test_model"
        assert len(model.components) == 2
        assert model.baseline_weights == ["astro_weight", "atmo_weight"]
    
    def test_required_variables(self, sample_components):
        """Test required variables property."""
        model = Model("test_model", sample_components, ["weight1", "weight2"])
        required_vars = model.required_variables
        
        # Should include variables from all components
        assert "log10_true_energy" in required_vars
    
    def test_exposed_parameters(self, sample_components):
        """Test exposed parameters property."""
        model = Model("test_model", sample_components, ["weight1", "weight2"])
        exposed_params = model.exposed_parameters
        
        # Should include parameters from all components
        expected_params = {"flux_norm", "spectral_index"}
        assert exposed_params == expected_params
    
    def test_parameter_mapping(self, sample_components):
        """Test parameter mapping property."""
        model = Model("test_model", sample_components, ["weight1", "weight2"])
        param_mapping = model.parameter_mapping
        
        assert "astro" in param_mapping
        assert "atmo" in param_mapping
        assert "powerlaw" in param_mapping["astro"]
        assert "norm2" in param_mapping["atmo"]
    
    def test_evaluate_single_component(self):
        """Test model evaluation with single component."""
        # Create a simple model with one component
        norm_factor = FluxNorm(name="norm")
        component = ModelComponent("test_comp", [norm_factor])
        model = Model("test_model", [component], ["baseline_weight"])
        
        input_variables = {"dummy": np.array([1, 2, 3])}
        parameters = {
            "flux_norm": 2.0,
            "baseline_weight": 3.0
        }
        
        result = model.evaluate(input_variables, parameters)
        
        # Result should be baseline_weight * component_evaluation = 3.0 * 2.0 = 6.0
        expected = np.array([6.0, 6.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_multiple_components(self, sample_components):
        """Test model evaluation with multiple components."""
        model = Model("test_model", sample_components, ["astro_weight", "atmo_weight"])
        
        input_variables = {
            "log10_true_energy": np.array([5.0])  # Single energy for simplicity
        }
        parameters = {
            "flux_norm": 1.0,
            "spectral_index": -2.0,
            "astro_weight": 2.0,
            "atmo_weight": 3.0
        }
        
        result = model.evaluate(input_variables, parameters)
        
        # Result should be sum of weighted components
        assert result.shape == (1,)
        assert result[0] > 0
    
    def test_evaluate_with_buffer_manager(self, sample_components):
        """Test model evaluation with buffer manager."""
        from pyForwardFolding.buffers import BufferManager
        
        model = Model("test_model", sample_components, ["astro_weight", "atmo_weight"])
        
        # Create datasets
        datasets = {
            "astro": {"log10_true_energy": np.array([5.0])},
            "atmo": {"log10_true_energy": np.array([5.0])}
        }
        
        # Create buffer manager
        buffer_manager = BufferManager.from_datasets(datasets)
        
        parameters = {
            "flux_norm": 1.0,
            "spectral_index": -2.0,
            "astro_weight": 2.0,
            "atmo_weight": 3.0
        }
        
        result = model.evaluate(datasets, parameters, buffer_manager=buffer_manager)
        
        # Should return result for each component
        assert len(result) == 2  # Two components
        assert "astro" in result
        assert "atmo" in result
        assert result["astro"].shape == (1,)
        assert result["atmo"].shape == (1,)


class TestModelIntegration:
    """Test integration between Model and ModelComponent."""
    
    def test_complex_model_evaluation(self):
        """Test evaluation of a more complex model."""
        # Create factors with parameter mappings
        powerlaw = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18,
            param_mapping={
                "flux_norm": "astro_norm",
                "spectral_index": "astro_index"
            }
        )
        
        atmo_norm = FluxNorm(
            name="atmo_norm",
            param_mapping={"flux_norm": "atmo_norm"}
        )
        
        # Create components
        astro_component = ModelComponent("astro", [powerlaw])
        atmo_component = ModelComponent("atmo", [atmo_norm])
        
        # Create model
        model = Model.from_pairs("full_model", [
            ("astro_weight", astro_component),
            ("atmo_weight", atmo_component)
        ])
        
        # Test evaluation
        input_variables = {
            "log10_true_energy": np.array([4.0, 5.0, 6.0])
        }
        parameters = {
            "astro_norm": 1.5,
            "astro_index": -2.5,
            "atmo_norm": 0.8,
            "astro_weight": 2.0,
            "atmo_weight": 1.0
        }
        
        result = model.evaluate(input_variables, parameters)
        
        # Check that result has correct shape and is positive
        assert result.shape == (3,)
        assert np.all(result > 0)
        
        # Check that exposed parameters are correct
        expected_exposed = {"astro_norm", "astro_index", "atmo_norm"}
        assert model.exposed_parameters == expected_exposed
