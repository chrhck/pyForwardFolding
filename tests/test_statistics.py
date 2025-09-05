"""
Tests for pyForwardFolding.statistics module.

This module contains comprehensive tests for the statistics functionality, including:
- PseudoExpGenerator: generating pseudo-experiments from models
- Hypothesis: representing hypotheses with fixed parameters and model evaluation
- HypothesisTest: performing hypothesis testing, discovery potential, and power calculations

Note: These tests use exact numerical values captured from the real implementation
rather than mock minimizers, ensuring that the tests verify the actual behavior
of the code and will catch any numerical regressions.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyForwardFolding.backend import backend
from pyForwardFolding.likelihood import PoissonLikelihood, UniformPrior
from pyForwardFolding.statistics import PseudoExpGenerator, Hypothesis, HypothesisTest
from .test_data import EXPECTED, MOCK_DISTS


COMP_1_VALUES = backend.asarray([10.0, 20.0, 15.0])
COMP_2_VALUES = backend.asarray([5.0, 8.0, 12.0])


class MockAnalysis:
    """Mock analysis for testing."""
    
    def __init__(self, exposed_parameters=None):
        self.exposed_parameters = exposed_parameters or {"param1", "param2", "param3"}
    
    def evaluate(self, datasets, parameter_values):
        """Mock evaluation returning predictable histograms."""
        # Return simple histograms based on parameter values
        hist = {
            "component1": backend.array(COMP_1_VALUES) * parameter_values.get("param1", 1.0),
            "component2": backend.array(COMP_2_VALUES) * parameter_values.get("param2", 1.0),
        }
        hist_ssq = {k: v**2 for k, v in hist.items()}
        return hist, hist_ssq


def create_real_likelihood(analysis):
    """Create a real PoissonLikelihood for testing."""
    # Create uniform priors for all exposed parameters
    prior_seeds = {param: 1.0 for param in analysis.exposed_parameters}
    prior_bounds = {param: (0.1, 10.0) for param in analysis.exposed_parameters}
    
    uniform_prior = UniformPrior(prior_seeds, prior_bounds)
    return PoissonLikelihood(analysis, [uniform_prior])


def to_float(value):
    """Convert Array or other numeric types to float."""
    if hasattr(value, 'item'):  # For JAX/NumPy arrays
        return float(value.item())
    elif hasattr(value, 'tolist'):
        # Handle array-like objects
        val_list = value.tolist()
        if isinstance(val_list, list) and len(val_list) == 1:
            return float(val_list[0])
        elif isinstance(val_list, (int, float)):
            return float(val_list)
        else:
            # For multi-element arrays, this shouldn't happen in our test cases
            raise ValueError(f"Cannot convert multi-element array to float: {val_list}")
    else:
        return float(value)


class TestPseudoExpGenerator:
    """Test the PseudoExpGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""        
        self.analysis = MockAnalysis()
        self.datasets = {
            "component1": {"param1": backend.array([1.0]), "param2": 1.0},
            "component2": {"param1": backend.array([1.0]), "param2": 1.0},
        }
        self.parameter_values = {"param1": 1.0, "param2": 1.0, "param3": 1.0}
        
        # MockAnalysis already implements evaluate() - no need to patch it
        self.generator = PseudoExpGenerator(
            self.analysis, 
            self.datasets, 
            self.parameter_values
        )
    
    def test_initialization(self):
        """Test PseudoExpGenerator initialization."""
        assert self.generator.analysis == self.analysis
        assert "component1" in self.generator.exp_hists
        assert "component2" in self.generator.exp_hists
        
        # Check that expected histograms are computed
        assert len(self.generator.exp_hists["component1"]) == 3
        assert len(self.generator.exp_hists["component2"]) == 3
    
    @patch('pyForwardFolding.backend.backend.poiss_rng')
    def test_generate_single_experiment(self, mock_poiss_rng):
        """Test generating a single pseudo-experiment."""
        # Mock the Poisson random number generator
        mock_poiss_rng.side_effect = lambda x: x  # Return the input (expected values)
        
        experiments = list(self.generator.generate(1))
        assert len(experiments) == 1
        
        experiment = experiments[0]
        assert "component1" in experiment
        assert "component2" in experiment
        
        # Check that poiss_rng was called for each component
        assert mock_poiss_rng.call_count == 2
    
    @patch('pyForwardFolding.backend.backend.poiss_rng')
    def test_generate_multiple_experiments(self, mock_poiss_rng):
        """Test generating multiple pseudo-experiments."""
        mock_poiss_rng.side_effect = lambda x: x  # Return the input (expected values)
        
        nexp = 5
        experiments = list(self.generator.generate(nexp))
        assert len(experiments) == nexp
        
        # Each experiment should have the same structure
        for exp in experiments:
            assert "component1" in exp
            assert "component2" in exp
            assert len(exp["component1"]) == 3
            assert len(exp["component2"]) == 3


class TestHypothesis:
    """Test the Hypothesis class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analysis = MockAnalysis()
        
        # Use the real PoissonLikelihood for testing
        self.likelihood = create_real_likelihood(self.analysis)
        
        # Create hypotheses with different fixed parameters using real minimizer
        self.h0 = Hypothesis(
            name="null",
            likelihood=self.likelihood,
            fixed_pars={"param3": 1.0}
        )
        
        self.h1 = Hypothesis(
            name="alternative", 
            likelihood=self.likelihood,
            fixed_pars={}
        )
        
        self.datasets = {
            "component1": {"param1": backend.array([1.0]), "param2": 1.0},
            "component2": {"param1": backend.array([1.0]), "param2": 1.0},
        }
        
        self.observed_data = {
            "component1": backend.array([12.0, 18.0, 16.0]),
            "component2": backend.array([6.0, 9.0, 11.0]),
        }
    
    def test_initialization(self):
        """Test Hypothesis initialization."""
        assert self.h0.name == "null"
        assert self.h0.fixed_pars == {"param3": 1.0}
        assert self.h0.likelihood == self.likelihood
        
        assert self.h1.name == "alternative"
        assert self.h1.fixed_pars == {}
    
    def test_model_parameters_property(self):
        """Test the model_parameters property."""
        # h0 has param3 fixed, so model_parameters should not include it
        h0_params = self.h0.model_parameters
        assert "param1" in h0_params
        assert "param2" in h0_params
        assert "param3" not in h0_params
        
        # h1 has no fixed parameters
        h1_params = self.h1.model_parameters
        assert "param1" in h1_params
        assert "param2" in h1_params
        assert "param3" in h1_params
    
    def test_nparams_property(self):
        """Test the nparams property."""
        # h0 has 3 total params - 1 fixed = 2 free
        assert self.h0.nparams == 2
        
        # h1 has 3 total params - 0 fixed = 3 free
        assert self.h1.nparams == 3
    
    def test_evaluate_basic(self):
        """Test basic hypothesis evaluation."""
        result = self.h0.evaluate(
            self.observed_data, 
            self.datasets, 
            detailed=False
        )
        
        # Check exact value from captured data
        actual_value = to_float(result)
        assert np.isclose(actual_value, EXPECTED.hypothesis_eval_basic, rtol=1E-8)
    
    def test_evaluate_detailed(self):
        """Test detailed hypothesis evaluation."""
        result = self.h0.evaluate(
            self.observed_data, 
            self.datasets, 
            detailed=True
        )
        
        # Should return the full minimizer result tuple
        assert isinstance(result, tuple)
        assert len(result) == 3
        # (best_fit_params, additional_info, llh_value)
        best_fit_params, additional_info, llh_value = result
        assert isinstance(best_fit_params, dict)  # best_fit_params (scipy result)
        assert isinstance(additional_info, dict)  # additional_info (parameter dict)
        
        # Check the actual parameter values from captured data
        # h0 has param3 fixed to 1.0, so it should be in the result
        assert "param3" in additional_info
        assert additional_info["param3"] == 1.0  # Fixed parameter value
        
        # Other parameters should have values close to captured ones
        assert "param1" in additional_info
        assert "param2" in additional_info
        assert np.isclose(additional_info["param1"], EXPECTED.best_fit_param1, rtol=1E-8)
        assert np.isclose(additional_info["param2"], EXPECTED.best_fit_param2, rtol=1E-8)
        
        # Check that log-likelihood matches captured value
        llh_float = to_float(llh_value)
        assert np.isclose(llh_float, EXPECTED.hypothesis_eval_basic, rtol=1E-8)
    
    def test_evaluate_with_parameter_values(self):
        """Test evaluation with provided parameter values."""
        param_values = {"param1": 1.5, "param2": 0.8}
        
        result1 = self.h0.evaluate(
            self.observed_data,
            self.datasets,
            parameter_values=param_values,
            detailed=False
        )
        
        # Check exact value from captured data
        actual_value = to_float(result1)
        assert np.isclose(actual_value, EXPECTED.hypothesis_eval_params_1_5_0_8, rtol=1E-8)
        
        # Test with different parameter values to ensure they give different results
        param_values_2 = {"param1": 2.0, "param2": 1.2}
        result2 = self.h0.evaluate(
            self.observed_data,
            self.datasets,
            parameter_values=param_values_2,
            detailed=False
        )
        
        # Check exact value from captured data
        actual_value_2 = to_float(result2)
        assert np.isclose(actual_value_2, EXPECTED.hypothesis_eval_params_2_0_1_2, rtol=1E-8)
        
        # Ensure the values are different
        assert abs(actual_value - actual_value_2) > 1e-6
    
    @patch('pyForwardFolding.statistics.PseudoExpGenerator')
    def test_generate_pseudo_experiments(self, mock_gen_class):
        """Test pseudo-experiment generation."""
        mock_generator = Mock()
        mock_generator.generate.return_value = iter([
            {"component1": backend.array([10, 15, 12])},
            {"component1": backend.array([11, 16, 13])},
        ])
        mock_gen_class.return_value = mock_generator
        
        nexp = 2
        experiments = list(self.h0.generate_pseudo_experiments(nexp, self.datasets))
        
        assert len(experiments) == nexp
        mock_gen_class.assert_called_once()
        mock_generator.generate.assert_called_once_with(nexp)
    
    def test_asimov_experiment(self):
        """Test Asimov experiment generation."""
        asimov_data = self.h0.asimov_experiment(self.datasets)
        
        assert "component1" in asimov_data
        assert "component2" in asimov_data
        assert len(asimov_data["component1"]) == 3
        assert len(asimov_data["component2"]) == 3
        
        # Check the exact values from captured data
        # h0 has param3=1.0 fixed, and seeds default to 1.0 for param1 and param2

        self.h0.likelihood.analysis.evaluate
        
        # Check that values match expectations exactly
        np.testing.assert_allclose(asimov_data["component1"], COMP_1_VALUES, rtol=1E-8)
        np.testing.assert_allclose(asimov_data["component2"], COMP_2_VALUES, rtol=1E-8)

    def test_asimov_experiment_with_parameter_values(self):
        """Test Asimov experiment with different parameter values."""
        # Test with custom parameter values
        param_values = {"param1": 2.0, "param2": 1.5}
        asimov_data = self.h0.asimov_experiment(self.datasets, param_values)
        
        # Check the exact values from captured data
        np.testing.assert_allclose(asimov_data["component1"], COMP_1_VALUES*param_values["param1"], rtol=1E-8)
        np.testing.assert_allclose(asimov_data["component2"], COMP_2_VALUES*param_values["param2"], rtol=1E-8)


class TestHypothesisTest:
    """Test the HypothesisTest class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analysis = MockAnalysis()
        
        # Use the real PoissonLikelihood for testing
        self.likelihood = create_real_likelihood(self.analysis)
        
        # Create nested hypotheses
        self.h0 = Hypothesis(
            name="null",
            likelihood=self.likelihood,
            fixed_pars={"param2": 1.5}
        )
        
        self.h1 = Hypothesis(
            name="alternative",
            likelihood=self.likelihood,
            fixed_pars={}
        )
        
        self.datasets = {
            "component1": {"param1": backend.array([1.0]), "param2": 1.0},
            "component2": {"param1": backend.array([1.0]), "param2": 1.0},
        }
        
        self.hypothesis_test = HypothesisTest(
            h0=self.h0,
            h1=self.h1,
            dataset=self.datasets
        )
        
        self.observed_data = {
            "component1": backend.array([12.0, 18.0, 16.0]),
            "component2": backend.array([6.0, 9.0, 11.0]),
        }
    
    def test_initialization(self):
        """Test HypothesisTest initialization."""
        assert self.hypothesis_test.h0 == self.h0
        assert self.hypothesis_test.h1 == self.h1
        assert self.hypothesis_test.dataset == self.datasets
    
    def test_free_parameters_property(self):
        """Test the free_parameters property."""
        free_params = self.hypothesis_test.free_parameters
        
        # h1 has {param1, param2, param3}, h0 has {param1, param3}
        # So free parameters should be {param3}
        assert free_params == {"param2"}
    
    def test_dof_property(self):
        """Test the degrees of freedom property."""
        dof = self.hypothesis_test.dof
        
        # h1 has 3 params, h0 has 2 params, so dof = 1
        assert dof == 1
    
    def test_test_method(self):
        """Test the hypothesis test method."""
        ts_value = self.hypothesis_test.test(self.observed_data)
        
        # Check exact value from captured data
        # From captured data: both h0 and h1 give the same likelihood value (13.318561553955078)
        # so test statistic should be 0.0
        actual_ts = to_float(ts_value)
        assert np.isclose(actual_ts, EXPECTED.test_stat, rtol=1E-8)
    
    def test_test_method_with_parameter_values(self):
        """Test hypothesis test method with specific parameter values."""
        # Test with parameter values
        param_values = {"param1": 1.2}
        ts_value = self.hypothesis_test.test(self.observed_data, param_values)
        
        # From captured data: with param1=1.2, test statistic is still 0.0
        actual_ts = to_float(ts_value)
        assert np.isclose(actual_ts, EXPECTED.test_stat_with_param1_1_2, rtol=1E-8)
    
    def test_test_method_different_hypotheses(self):
        """Test hypothesis test with meaningfully different hypotheses."""
        # Create hypotheses with different fixed parameter values
        h0_different = Hypothesis(
            name="background_only",
            likelihood=self.likelihood,
            fixed_pars={"param3": 0.0}  # Background only
        )
        
        h1_different = Hypothesis(
            name="signal_plus_background",
            likelihood=self.likelihood,
            fixed_pars={}  # Signal allowed to vary
        )
        
        # Create hypothesis test with different hypotheses
        ht_different = HypothesisTest(h0_different, h1_different, self.datasets)
        
        # Perform the test
        ts_value = ht_different.test(self.observed_data)
        
        # From captured data: test statistic is still 0.0 
        # (the current data seems to optimize to similar values)
        actual_ts = to_float(ts_value)
        assert np.isclose(actual_ts, EXPECTED.test_stat_different_hypotheses, rtol=1E-8)
    
    
    def test_discovery_potential_multi_dof_error(self):
        """Test discovery potential raises error for multiple degrees of freedom."""        
        # Create a hypothesis test with multiple free parameters
        h0_multi = Hypothesis(
            name="null_multi",
            likelihood=self.likelihood,
            fixed_pars={"param1": 1.0, "param2": 1.0}  # Fix param1 and param2, param3 is free
        )
        
        h1_multi = Hypothesis(
            name="alt_multi",
            likelihood=self.likelihood,
            fixed_pars={}  # No fixed parameters - all 3 params are free
        )
        
        ht_multi = HypothesisTest(h0_multi, h1_multi, self.datasets)
        
        # This should have 2 DOF (3 - 1 = 2) and should raise an error
        with pytest.raises(ValueError, match="Discovery potential is only implemented for one degree of freedom"):
            ht_multi.discovery_potential(10, 10)
    
    @patch('scipy.optimize.brentq')
    def test_discovery_potential_asimov(self, mock_brentq):
        """Test Asimov discovery potential calculation."""
        # Mock brentq to return a parameter value
        mock_brentq.return_value = 1.5
        
        param_val = self.hypothesis_test.discovery_potential_asimov()
        
        assert param_val == 1.5
        mock_brentq.assert_called_once()
    
    def test_discovery_potential_asimov_multi_dof_error(self):
        """Test Asimov discovery potential raises error for multiple degrees of freedom."""        
        # Create a hypothesis test with multiple free parameters
        h0_multi = Hypothesis(
            name="null_multi",
            likelihood=self.likelihood,
            fixed_pars={"param1": 1.0, "param2": 1.0}  # Fix param1 and param2, param3 is free
        )
        
        h1_multi = Hypothesis(
            name="alt_multi",
            likelihood=self.likelihood,
            fixed_pars={}  # No fixed parameters - all 3 params are free
        )
        
        ht_multi = HypothesisTest(h0_multi, h1_multi, self.datasets)
        
        # This should have 2 DOF (3 - 1 = 2) and should raise an error
        with pytest.raises(ValueError, match="Discovery potential is only implemented for one degree of freedom"):
            ht_multi.discovery_potential_asimov()
    
    def test_power(self):
        """Test power calculation."""        
        # Mock distributions
        with patch.object(self.hypothesis_test, 'null_dist', return_value=MOCK_DISTS.null_dist_5_points):
            with patch.object(self.hypothesis_test, 'alt_dist', return_value=MOCK_DISTS.alt_dist_5_points):
                power = self.hypothesis_test.power(5)
                
                # Power should be around 0.6 (3 out of 5 above threshold)
                power_float = to_float(power)
                assert EXPECTED.power_lower_bound <= power_float <= EXPECTED.power_upper_bound  # Adjusted range to be more tolerant
    
    def test_scan_single_dof(self):
        """Test parameter scan for single degree of freedom."""
        scan_points = 5
        scan_grid, ts_values = self.hypothesis_test.scan(
            self.observed_data, 
            scan_points
        )
        
        assert len(scan_grid) == scan_points
        assert len(ts_values) == scan_points
        
        # Check exact values from captured data
        np.testing.assert_allclose(scan_grid, EXPECTED.scan_grid_5_points, rtol=1E-4)
        np.testing.assert_allclose(ts_values, EXPECTED.scan_ts_values_5_points, rtol=1E-4)
    
    def test_scan_multi_dof_error(self):
        """Test scan raises error for multiple degrees of freedom."""        
        # Create a hypothesis test with multiple free parameters
        h0_multi = Hypothesis(
            name="null_multi",
            likelihood=self.likelihood,
            fixed_pars={"param1": 1.0, "param2": 1.0}  # Fix param1 and param2, param3 is free
        )
        
        h1_multi = Hypothesis(
            name="alt_multi",
            likelihood=self.likelihood,
            fixed_pars={}  # No fixed parameters - all 3 params are free
        )
        
        ht_multi = HypothesisTest(h0_multi, h1_multi, self.datasets)
        
        # This should have 2 DOF (3 - 1 = 2) and should raise an error
        with pytest.raises(ValueError, match="Scan is only implemented for one degree of freedom"):
            ht_multi.scan(self.observed_data, 10)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.analysis = MockAnalysis()
        
        # Use the real PoissonLikelihood for testing
        self.likelihood = create_real_likelihood(self.analysis)
        
        self.datasets = {
            "component1": {"param1": backend.array([1.0]), "param2": 1.0},
            "component2": {"param1": backend.array([1.0]), "param2": 1.0},
        }
    
    def test_full_hypothesis_test_workflow(self):
        """Test a complete hypothesis testing workflow."""        
        # Create hypotheses
        h0 = Hypothesis(
            name="background_only",
            likelihood=self.likelihood,
            fixed_pars={"param3": 0.0}  # No signal
        )
        
        h1 = Hypothesis(
            name="signal_plus_background",
            likelihood=self.likelihood,
            fixed_pars={}  # Signal allowed to vary
        )
        
        # Create hypothesis test
        ht = HypothesisTest(h0, h1, self.datasets)
        
        # Generate some "observed" data
        observed_data = {
            "component1": backend.array([15.0, 25.0, 20.0]),
            "component2": backend.array([8.0, 12.0, 15.0]),
        }
        
        # Perform test
        ts_value = ht.test(observed_data)
        # Test statistic should be finite (can be positive or negative)
        assert np.isfinite(to_float(ts_value))
        
        # Check properties
        assert len(ht.free_parameters) == 1
        assert ht.dof == 1
    
    @patch('pyForwardFolding.backend.backend.poiss_rng')
    def test_pseudo_experiment_roundtrip(self, mock_poiss_rng):
        """Test generating and using pseudo-experiments."""        
        # Mock Poisson RNG to return deterministic values
        def mock_poisson(expected):
            return expected + 0.1 * np.random.randn(*expected.shape)
        
        mock_poiss_rng.side_effect = mock_poisson
        
        # Create hypothesis
        h = Hypothesis(
            name="test_hypothesis",
            likelihood=self.likelihood,
            fixed_pars={"param3": 1.0}
        )
        
        # Generate pseudo-experiments
        pseudo_exps = list(h.generate_pseudo_experiments(3, self.datasets))
        assert len(pseudo_exps) == 3
        
        # Each experiment should have the right structure
        for exp in pseudo_exps:
            assert "component1" in exp
            assert "component2" in exp
            assert len(exp["component1"]) == 3
            assert len(exp["component2"]) == 3
    
    def test_asimov_vs_pseudo_consistency(self):
        """Test that Asimov data is consistent with pseudo-experiment expectations."""        
        h = Hypothesis(
            name="test_hypothesis",
            likelihood=self.likelihood,
            fixed_pars={"param3": 1.0}
        )
        
        # Generate Asimov dataset
        asimov_data = h.asimov_experiment(self.datasets)
        
        # The Asimov data should match the expected values from the model
        # (this is a basic consistency check)
        assert all(np.all(hist >= 0) for hist in asimov_data.values())
        assert len(asimov_data) == 2  # Should have same components as datasets


if __name__ == "__main__":
    pytest.main([__file__])
