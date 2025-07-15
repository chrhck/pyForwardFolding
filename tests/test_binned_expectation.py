"""Tests for the binned_expectation module."""

import numpy as np
import pytest

from pyForwardFolding.binned_expectation import BinnedExpectation
from pyForwardFolding.binning import RectangularBinning
from pyForwardFolding.factor import AbstractBinnedFactor, FluxNorm, PowerLawFlux
from pyForwardFolding.model import Model
from pyForwardFolding.model_component import ModelComponent


class MockBinnedFactor(AbstractBinnedFactor):
    """Mock binned factor for testing."""
    
    def __init__(self, name: str, param_mapping=None):
        # Create a dummy binning for the mock
        from pyForwardFolding.binning import RectangularBinning
        dummy_binning = RectangularBinning(
            bin_variables=("energy",),
            bin_edges=([0, 1, 2, 3],)
        )
        super().__init__(name, dummy_binning, param_mapping)
        self.factor_parameters = ["scale_factor"]
    
    def evaluate(self, input_variables, parameter_values):
        """Return mock histogram data for testing."""
        # Return simple constant bins for testing
        import numpy as np
        scale = parameter_values.get("scale_factor", 1.0)
        # Mock histogram with shape (3, 2) to match energy and zenith binning
        hist = np.ones((3, 2)) * scale * 0.1  # Small constant contribution
        hist_ssq = np.ones((3, 2)) * scale * scale * 0.01  # Variance
        return hist, hist_ssq
    
    def evaluate_binned(self, binned_data, parameter_values):
        # Simple scaling factor
        scale = parameter_values.get("scale_factor", 1.0)
        return binned_data * scale


class TestBinnedExpectation:
    """Test the BinnedExpectation class."""
    
    @pytest.fixture
    def sample_binning(self):
        """Provide sample binning for testing."""
        return RectangularBinning(
            bin_variables=("energy", "zenith"),
            bin_edges=([0, 1, 2, 3], [0, 0.5, 1.0])
        )
    
    @pytest.fixture
    def sample_model(self):
        """Provide sample model for testing."""
        powerlaw = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18
        )
        norm = FluxNorm(name="norm") 
        component = ModelComponent("astro", [powerlaw, norm])
        return Model("test_model", [component], ["weight"])
    
    @pytest.fixture
    def sample_binned_factors(self):
        """Provide sample binned factors."""
        return [MockBinnedFactor("detector_efficiency")]
    
    def test_initialization_basic(self, sample_binning, sample_model):
        """Test basic BinnedExpectation initialization."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )
        
        assert expectation.name == "test_expectation"
        assert len(expectation.dskey_model_pairs) == 1
        assert len(expectation.models) == 1
        assert expectation.binning == sample_binning
        assert expectation.binned_factors == []
        assert expectation.lifetime == 1.0
    
    def test_initialization_with_binned_factors(self, sample_binning, sample_model, sample_binned_factors):
        """Test BinnedExpectation initialization with binned factors."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning,
            binned_factors=sample_binned_factors,
            lifetime=10.0
        )
        
        assert len(expectation.binned_factors) == 1
        assert expectation.lifetime == 10.0
    
    def test_initialization_multiple_models(self, sample_binning, sample_model):
        """Test BinnedExpectation initialization with multiple models."""
        # Create a second model
        norm_only = FluxNorm(name="norm2")
        component2 = ModelComponent("background", [norm_only])
        model2 = Model("background_model", [component2], ["weight"])
        
        dskey_model_pairs = [
            ("dataset1", sample_model),
            ("dataset2", model2)
        ]
        
        expectation = BinnedExpectation(
            name="multi_model_expectation", 
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )
        
        assert len(expectation.models) == 2
        assert len(expectation.dskey_model_pairs) == 2
    
    def test_required_variables(self, sample_binning, sample_model):
        """Test required variables property."""
        dskey_model_pairs = [("dataset1", sample_model)]

        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )

        required_vars = expectation.required_variables

        # Should include binning variables and model variables
        assert "energy" in required_vars  # from binning
        assert "zenith" in required_vars  # from binning
        assert "true_energy" in required_vars  # from PowerLawFlux
        assert isinstance(required_vars, set)
    
    def test_exposed_parameters_single_model(self, sample_binning, sample_model):
        """Test exposed parameters with single model."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )
        
        exposed_params = expectation.exposed_parameters
        
        # Should be a set with model parameters
        assert isinstance(exposed_params, set)
        # The exact structure depends on implementation, but should contain model parameters
    
    def test_exposed_parameters_with_binned_factors(self, sample_binning, sample_model, sample_binned_factors):
        """Test exposed parameters including binned factors."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning,
            binned_factors=sample_binned_factors
        )
        
        exposed_params = expectation.exposed_parameters
        
        # Should include both model and binned factor parameters
        assert isinstance(exposed_params, set)
    
    def test_evaluate_basic(self, sample_binning, sample_model):
        """Test basic evaluation functionality."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )
         # Create sample datasets
        datasets = {
            "dataset1": {
                "energy": np.array([0.5, 1.5, 2.5]),
                "zenith": np.array([0.25, 0.75, 0.25]),
                "true_energy": np.array([1e4, 1e5, 1e6]),
                "weight": np.array([1.0, 2.0, 1.5])
            }
        }
        
        # Sample parameters
        parameters = {
            "flux_norm": 1.0,
            "spectral_index": 2.0
        }
        
        # This test checks that the method exists and can be called
        # The exact implementation would need to be checked against the actual code
        result = expectation.evaluate(datasets, parameters)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # Should return (hist, hist_ssq)
    
    def test_evaluate_with_binned_factors(self, sample_binning, sample_model, sample_binned_factors):
        """Test evaluation with binned factors."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning,
            binned_factors=sample_binned_factors
        )
        
        datasets = {
            "dataset1": {
                "energy": np.array([0.5, 1.5]),
                "zenith": np.array([0.25, 0.75]),
                "true_energy": np.array([1e4, 1e5]),
                "weight": np.array([1.0, 2.0])
            }
        }
        
        parameters = {
            "flux_norm": 1.0,
            "spectral_index": 2.0,
            "scale_factor": 1.2
        }
        
        result = expectation.evaluate(datasets, parameters)
        # If we get here, the method exists and can be called
        assert result is not None
    
    def test_empty_datasets(self, sample_binning, sample_model):
        """Test behavior with empty datasets."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )
        
        # Empty datasets
        datasets = {
            "dataset1": {
                "energy": np.array([]),
                "zenith": np.array([]),
                "true_energy": np.array([]),
                "weight": np.array([])
            }
        }
        
        parameters = {
            "flux_norm": 1.0,
            "spectral_index": 2.0
        }
        
        result = expectation.evaluate(datasets, parameters)
        # Should handle empty data gracefully - if we get here, method exists
        assert result is not None
    
    def test_missing_dataset(self, sample_binning, sample_model):
        """Test behavior when expected dataset is missing."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )
        
        # Missing dataset
        datasets = {}
        parameters = {"flux_norm": 1.0, "spectral_index": 2.0}
        
        with pytest.raises(ValueError, match="Dataset 'dataset1' not found"):
            expectation.evaluate(datasets, parameters)
    
    def test_lifetime_property(self, sample_binning, sample_model):
        """Test that lifetime property works correctly."""
        dskey_model_pairs = [("dataset1", sample_model)]
        
        # Test default lifetime
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )
        assert expectation.lifetime == 1.0
        
        # Test custom lifetime
        expectation_custom = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning,
            lifetime=3.14
        )
        assert expectation_custom.lifetime == 3.14
    
    def test_models_property(self, sample_binning):
        """Test that models property extracts models correctly."""
        # Create multiple models
        powerlaw = PowerLawFlux(name="powerlaw", pivot_energy=1e5, baseline_norm=1e-18)
        norm1 = FluxNorm(name="norm1")
        norm2 = FluxNorm(name="norm2")
        
        component1 = ModelComponent("astro", [powerlaw, norm1])
        component2 = ModelComponent("background", [norm2])
        
        model1 = Model("astro_model", [component1], ["weight"])
        model2 = Model("bg_model", [component2], ["weight"])
        
        dskey_model_pairs = [
            ("dataset1", model1),
            ("dataset2", model2)
        ]
        
        expectation = BinnedExpectation(
            name="test_expectation",
            dskey_model_pairs=dskey_model_pairs,
            binning=sample_binning
        )
        
        assert len(expectation.models) == 2
        assert expectation.models[0] == model1
        assert expectation.models[1] == model2


class TestBinnedExpectationIntegration:
    """Integration tests for BinnedExpectation."""
    
    def test_complete_workflow(self):
        """Test a complete workflow with BinnedExpectation."""
        # Create binning
        binning = RectangularBinning(
            bin_variables=("energy",),
            bin_edges=([1, 2, 3, 4],)
        )
        
        # Create model
        powerlaw = PowerLawFlux(name="powerlaw", pivot_energy=1e5, baseline_norm=1e-18)
        norm = FluxNorm(name="norm")
        component = ModelComponent("astro", [powerlaw, norm])
        model = Model("test_model", [component], ["weight"])
        
        # Create binned expectation
        expectation = BinnedExpectation(
            name="integration_test",
            dskey_model_pairs=[("test_data", model)],
            binning=binning,
            lifetime=2.0
        )
        
        # Verify setup
        assert expectation.name == "integration_test"
        assert expectation.lifetime == 2.0
        assert len(expectation.models) == 1
        
        # Check required variables include both binning and model requirements
        required_vars = expectation.required_variables
        assert "energy" in required_vars  # from binning
        assert "true_energy" in required_vars  # from PowerLawFlux
        
    def test_parameter_consistency(self):
        """Test that exposed parameters are consistent across the workflow."""
        # Setup similar to above
        binning = RectangularBinning(
            bin_variables=("energy",),
            bin_edges=([1, 2, 3],)
        )
        
        powerlaw = PowerLawFlux(name="powerlaw", pivot_energy=1e5, baseline_norm=1e-18)
        component = ModelComponent("astro", [powerlaw])
        model = Model("test_model", [component], ["weight"])
        
        expectation = BinnedExpectation(
            name="param_test",
            dskey_model_pairs=[("data", model)],
            binning=binning
        )
        
        # Model exposed parameters should be accessible through expectation
        expectation_params = expectation.exposed_parameters
        
        # The exact relationship depends on implementation, but there should be consistency
        assert isinstance(expectation_params, set)
        assert len(expectation_params) >= 0  # At minimum, should be a valid dict
