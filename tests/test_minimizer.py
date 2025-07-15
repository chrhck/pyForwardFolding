"""Tests for the minimizer module."""

import numpy as np
import pytest

from pyForwardFolding.minimizer import (
    WrappedLLH,
    destructure_args,
    flat_index_dict_mapping,
    restructure_args,
)


class TestUtilityFunctions:
    """Test utility functions for parameter handling."""
    
    def test_flat_index_dict_mapping_simple(self):
        """Test basic flat index mapping."""
        exp_vars = {"param1", "param2", "param3"}
        
        mapping = flat_index_dict_mapping(exp_vars)
        
        # Should create a mapping for all variables
        assert len(mapping) == 3
        assert set(mapping.keys()) == exp_vars
        assert set(mapping.values()) == {0, 1, 2}
    
    def test_flat_index_dict_mapping_with_fixed(self):
        """Test flat index mapping with fixed parameters."""
        exp_vars = {"param1", "param2", "param3", "param4"}
        fixed_params = {"param2": 2.0, "param4": 4.0}
        
        mapping = flat_index_dict_mapping(exp_vars, fixed_params)
        
        # Should only map non-fixed parameters
        assert len(mapping) == 2
        assert "param1" in mapping
        assert "param3" in mapping
        assert "param2" not in mapping
        assert "param4" not in mapping
        assert set(mapping.values()) == {0, 1}
    
    def test_flat_index_dict_mapping_empty(self):
        """Test flat index mapping with empty variables."""
        exp_vars = set()
        
        mapping = flat_index_dict_mapping(exp_vars)
        
        assert len(mapping) == 0
        assert mapping == {}
    
    def test_flat_index_dict_mapping_all_fixed(self):
        """Test flat index mapping when all parameters are fixed."""
        exp_vars = {"param1", "param2"}
        fixed_params = {"param1": 1.0, "param2": 2.0}
        
        mapping = flat_index_dict_mapping(exp_vars, fixed_params)
        
        assert len(mapping) == 0
        assert mapping == {}
    
    def test_restructure_args_simple(self):
        """Test basic argument restructuring."""
        flat_args = [1.0, 2.0, 3.0]
        exp_vars = {"param1", "param2", "param3"}
        
        result = restructure_args(flat_args, exp_vars)
        
        # Should create dictionary with flat args assigned to variables
        assert len(result) == 3
        assert set(result.keys()) == exp_vars
        assert set(result.values()) == set(flat_args)
    
    def test_restructure_args_with_fixed(self):
        """Test argument restructuring with fixed parameters."""
        flat_args = [1.0, 3.0]
        exp_vars = {"param1", "param2", "param3"}
        fixed_params = {"param2": 2.0}
        
        result = restructure_args(flat_args, exp_vars, fixed_params)
        
        # Should include both flat args and fixed parameters
        assert len(result) == 3
        assert result["param2"] == 2.0
        assert "param1" in result
        assert "param3" in result
        assert result["param1"] in flat_args
        assert result["param3"] in flat_args
    
    def test_restructure_args_mismatched_length(self):
        """Test argument restructuring with mismatched lengths."""
        flat_args = [1.0, 2.0]  # Only 2 args
        exp_vars = {"param1", "param2", "param3"}  # But 3 variables
        
        # Should handle gracefully (behavior depends on implementation)
        # This might raise an error or handle it differently
        with pytest.raises(IndexError):
            restructure_args(flat_args, exp_vars)
    
    def test_destructure_args_simple(self):
        """Test basic argument destructuring."""
        parameter_dict = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
        exp_vars = {"param1", "param2", "param3"}
        
        flat_args = destructure_args(parameter_dict, exp_vars, {})
        
        # Should return flat array with values in consistent order
        assert len(flat_args) == 3
        assert set(flat_args) == {1.0, 2.0, 3.0}
    
    def test_destructure_args_with_fixed(self):
        """Test argument destructuring with fixed parameters."""
        parameter_dict = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
        exp_vars = {"param1", "param2", "param3"}
        fixed_params = {"param2": 2.0}
        
        flat_args = destructure_args(parameter_dict, exp_vars, fixed_params)
        
        # Should only include non-fixed parameters
        assert len(flat_args) == 2
        assert 2.0 not in flat_args  # Fixed parameter should not be included
        assert set(flat_args) == {1.0, 3.0}
    
    def test_destructure_args_missing_parameter(self):
        """Test argument destructuring with missing parameters."""
        parameter_dict = {"param1": 1.0, "param2": 2.0}  # Missing param3
        exp_vars = {"param1", "param2", "param3"}
        
        with pytest.raises(KeyError):
            destructure_args(parameter_dict, exp_vars, {})
    
    def test_roundtrip_restructure_destructure(self):
        """Test that restructure and destructure are inverse operations."""
        original_params = {"param1": 1.5, "param2": 2.5, "param3": 3.5}
        exp_vars = {"param1", "param2", "param3"}
        fixed_params = {"param2": 2.5}
        
        # Destructure to flat args
        flat_args = destructure_args(original_params, exp_vars, fixed_params)
        
        # Restructure back to dictionary
        reconstructed_params = restructure_args(flat_args, exp_vars, fixed_params)
        
        # Should match original
        assert reconstructed_params == original_params


class MockLikelihood:
    """Mock likelihood for testing WrappedLLH."""
    
    def __init__(self, return_value=1.0):
        self.return_value = return_value
        self.call_count = 0
        self.last_call_args = None
    
    def llh(self, observed_data, datasets, parameters, empty_bins="skip"):
        self.call_count += 1
        self.last_call_args = (observed_data, datasets, parameters, empty_bins)
        return self.return_value


class TestWrappedLLH:
    """Test the WrappedLLH class."""
    
    def test_initialization(self):
        """Test WrappedLLH initialization."""
        mock_likelihood = MockLikelihood()
        observed_data = {"detector": np.array([1, 2, 3])}
        datasets = {"component": {"energy": np.array([1.0, 2.0, 3.0])}}
        fixed_params = {"fixed_param": 1.0}
        priors = []
        
        wrapped = WrappedLLH(
            mock_likelihood, observed_data, datasets, fixed_params, priors
        )
        
        assert wrapped.likelihood == mock_likelihood
        assert wrapped.observed_data == observed_data
        assert wrapped.datasets == datasets
        assert wrapped.fixed_params == fixed_params
        assert wrapped.priors == priors
    
    def test_call_basic(self):
        """Test basic call functionality."""
        mock_likelihood = MockLikelihood(return_value=5.0)
        observed_data = {"detector": np.array([1, 2, 3])}
        datasets = {"component": {"energy": np.array([1.0, 2.0, 3.0])}}
        
        wrapped = WrappedLLH(mock_likelihood, observed_data, datasets, {}, [])
        
        # Mock exposed parameters for the likelihood
        wrapped.likelihood.get_analysis = lambda: type('MockAnalysis', (), {
            'exposed_parameters': {"param1", "param2"}
        })()
        
        flat_args = [1.0, 2.0]
        result = wrapped(flat_args)
        
        # Should call likelihood and return its result
        assert result == 5.0
        assert mock_likelihood.call_count == 1
        
        # Check that likelihood was called with correct arguments
        obs_data, datasets_arg, params_arg, empty_bins = mock_likelihood.last_call_args
        assert obs_data == observed_data
        assert datasets_arg == datasets
        assert empty_bins == "skip"
    
    def test_call_with_fixed_params(self):
        """Test call with fixed parameters."""
        mock_likelihood = MockLikelihood(return_value=3.0)
        observed_data = {"detector": np.array([1, 2])}
        datasets = {"component": {"energy": np.array([1.0, 2.0])}}
        fixed_params = {"fixed_param": 42.0}
        
        wrapped = WrappedLLH(mock_likelihood, observed_data, datasets, fixed_params, [])
        
        # Mock exposed parameters
        wrapped.likelihood.get_analysis = lambda: type('MockAnalysis', (), {
            'exposed_parameters': {"param1", "fixed_param"}
        })()
        
        flat_args = [1.0]  # Only one free parameter
        result = wrapped(flat_args)
        
        assert result == 3.0
        
        # Check that fixed parameter was included
        _, _, params_arg, _ = mock_likelihood.last_call_args
        assert "fixed_param" in params_arg
        assert params_arg["fixed_param"] == 42.0
        assert "param1" in params_arg
    
    def test_call_with_priors(self):
        """Test call with priors."""
        from pyForwardFolding.likelihood import GaussianUnivariatePrior
        
        mock_likelihood = MockLikelihood(return_value=2.0)
        observed_data = {"detector": np.array([1])}
        datasets = {"component": {"energy": np.array([1.0])}}
        
        # Create a prior that adds 0.5 to the likelihood
        prior = GaussianUnivariatePrior("param1", mean=1.0, sigma=1.0)
        
        wrapped = WrappedLLH(mock_likelihood, observed_data, datasets, {}, [prior])
        
        # Mock exposed parameters
        wrapped.likelihood.get_analysis = lambda: type('MockAnalysis', (), {
            'exposed_parameters': {"param1"}
        })()
        
        flat_args = [2.0]  # One sigma away from prior mean
        result = wrapped(flat_args)
        
        # Should be likelihood + prior = 2.0 + 0.5 * (2-1)^2 = 2.5
        expected = 2.0 + 0.5 * ((2.0 - 1.0) / 1.0) ** 2
        assert abs(result - expected) < 1e-10
    
    def test_call_with_multiple_priors(self):
        """Test call with multiple priors."""
        from pyForwardFolding.likelihood import GaussianUnivariatePrior
        
        mock_likelihood = MockLikelihood(return_value=1.0)
        observed_data = {"detector": np.array([1])}
        datasets = {"component": {"energy": np.array([1.0])}}
        
        # Create multiple priors
        prior1 = GaussianUnivariatePrior("param1", mean=1.0, sigma=1.0)
        prior2 = GaussianUnivariatePrior("param2", mean=2.0, sigma=0.5)
        
        wrapped = WrappedLLH(
            mock_likelihood, observed_data, datasets, {}, [prior1, prior2]
        )
        
        # Mock exposed parameters
        wrapped.likelihood.get_analysis = lambda: type('MockAnalysis', (), {
            'exposed_parameters': {"param1", "param2"}
        })()
        
        flat_args = [1.5, 2.5]  # Half sigma and one sigma away from means
        result = wrapped(flat_args)
        
        # Should be likelihood + prior1 + prior2
        prior1_contrib = 0.5 * ((1.5 - 1.0) / 1.0) ** 2  # 0.125
        prior2_contrib = 0.5 * ((2.5 - 2.0) / 0.5) ** 2  # 0.5
        expected = 1.0 + prior1_contrib + prior2_contrib
        
        assert abs(result - expected) < 1e-10
    
    def test_call_empty_args(self):
        """Test call with empty arguments (all parameters fixed)."""
        mock_likelihood = MockLikelihood(return_value=4.0)
        observed_data = {"detector": np.array([1])}
        datasets = {"component": {"energy": np.array([1.0])}}
        fixed_params = {"param1": 1.0}
        
        wrapped = WrappedLLH(mock_likelihood, observed_data, datasets, fixed_params, [])
        
        # Mock exposed parameters (all fixed)
        wrapped.likelihood.get_analysis = lambda: type('MockAnalysis', (), {
            'exposed_parameters': {"param1"}
        })()
        
        flat_args = []  # No free parameters
        result = wrapped(flat_args)
        
        assert result == 4.0
        
        # Check that only fixed parameters were passed
        _, _, params_arg, _ = mock_likelihood.last_call_args
        assert params_arg == {"param1": 1.0}


class TestMinimizerIntegration:
    """Test integration of minimizer components."""
    
    def test_parameter_handling_workflow(self):
        """Test complete parameter handling workflow."""
        # Define a complete parameter set
        all_params = {
            "astro_norm": 1.0,
            "astro_index": -2.0,
            "atmo_norm": 0.5,
            "baseline_weight": 1.0
        }
        
        # Some parameters are fixed
        fixed_params = {"baseline_weight": 1.0}
        
        # Exposed parameters (what the analysis expects)
        exp_vars = {"astro_norm", "astro_index", "atmo_norm", "baseline_weight"}
        
        # Step 1: Create index mapping
        index_mapping = flat_index_dict_mapping(exp_vars, fixed_params)
        
        # Step 2: Destructure to flat args
        flat_args = destructure_args(all_params, exp_vars, fixed_params)
        
        # Step 3: Restructure back (simulating optimization step)
        reconstructed = restructure_args(flat_args, exp_vars, fixed_params)
        
        # Should match original
        assert reconstructed == all_params
        
        # Verify that only free parameters are in flat_args
        assert len(flat_args) == 3  # 4 total - 1 fixed
        assert len(index_mapping) == 3
    
    def test_wrapped_llh_optimization_interface(self):
        """Test that WrappedLLH provides correct interface for optimization."""
        mock_likelihood = MockLikelihood(return_value=1.0)
        observed_data = {"detector": np.array([5, 3, 2])}
        datasets = {"component": {"energy": np.array([1.0, 2.0, 3.0])}}
        fixed_params = {"fixed_param": 1.0}
        
        wrapped = WrappedLLH(mock_likelihood, observed_data, datasets, fixed_params, [])
        
        # Mock the analysis
        wrapped.likelihood.get_analysis = lambda: type('MockAnalysis', (), {
            'exposed_parameters': {"param1", "param2", "fixed_param"}
        })()
        
        # Test that wrapped function can be called with different parameter values
        test_points = [
            [1.0, 2.0],
            [1.5, 2.5],
            [0.5, 1.5]
        ]
        
        for point in test_points:
            result = wrapped(point)
            assert isinstance(result, (float, np.floating))
            assert np.isfinite(result)
        
        # Should have been called once for each test point
        assert mock_likelihood.call_count == len(test_points)
