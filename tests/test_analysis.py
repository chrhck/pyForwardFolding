"""Tests for the analysis module."""

import numpy as np
import pytest

from pyForwardFolding.analysis import Analysis
from pyForwardFolding.binned_expectation import BinnedExpectation
from pyForwardFolding.binning import RectangularBinning
from pyForwardFolding.factor import FluxNorm, PowerLawFlux
from pyForwardFolding.model import Model
from pyForwardFolding.model_component import ModelComponent


class TestAnalysis:
    """Test the Analysis class."""
    
    @pytest.fixture
    def sample_expectations(self):
        """Create sample binned expectations for testing."""
        # Create binning
        bin_edges = [np.array([0, 1, 2, 3])]
        binning = RectangularBinning(bin_edges, ["energy"])
        
        # Create simple model with FluxNorm
        norm_factor = FluxNorm(name="norm")
        component = ModelComponent("test_comp", [norm_factor])
        model = Model("test_model", [component], ["baseline_weight"])
        
        # Create first expectation
        expectation1 = BinnedExpectation("detector1", model, binning)
        
        # Create second expectation with different binning
        bin_edges2 = [np.array([0, 2, 4])]
        binning2 = RectangularBinning(bin_edges2, ["energy"])
        expectation2 = BinnedExpectation("detector2", model, binning2)
        
        return {"detector1": expectation1, "detector2": expectation2}
    
    def test_initialization(self, sample_expectations):
        """Test Analysis initialization."""
        analysis = Analysis(sample_expectations)
        
        assert analysis.expectations == sample_expectations
        assert len(analysis.expectations) == 2
        assert "detector1" in analysis.expectations
        assert "detector2" in analysis.expectations
    
    def test_required_variables(self, sample_expectations):
        """Test required variables property."""
        analysis = Analysis(sample_expectations)
        required_vars = analysis.required_variables
        
        # Both expectations use the same model requiring "energy" for binning
        # FluxNorm doesn't require any specific variables
        assert "energy" in required_vars
    
    def test_exposed_parameters(self, sample_expectations):
        """Test exposed parameters property."""
        analysis = Analysis(sample_expectations)
        exposed_params = analysis.exposed_parameters
        
        # Both expectations use the same model with FluxNorm
        expected_params = {"flux_norm"}
        assert exposed_params == expected_params
    
    def test_evaluate_single_expectation(self):
        """Test analysis evaluation with single expectation."""
        # Create simple analysis
        bin_edges = [np.array([0, 1, 2])]
        binning = RectangularBinning(bin_edges, ["energy"])
        
        norm_factor = FluxNorm(name="norm")
        component = ModelComponent("test_comp", [norm_factor])
        model = Model("test_model", [component], ["baseline_weight"])
        
        expectation = BinnedExpectation("detector", model, binning)
        analysis = Analysis({"detector": expectation})
        
        # Test datasets and parameters
        datasets = {
            "test_comp": {
                "energy": np.array([0.5])  # One event in first bin
            }
        }
        
        parameters = {
            "flux_norm": 2.0,
            "baseline_weight": 3.0
        }
        
        hist, hist_ssq = analysis.evaluate(datasets, parameters)
        
        # Check that we get results for our detector
        assert "detector" in hist
        assert "detector" in hist_ssq
        
        # Check histogram values
        detector_hist = hist["detector"]
        assert detector_hist.shape == (2,)  # 2 bins
        assert detector_hist[0] > 0  # First bin should have events
        assert detector_hist[1] == 0  # Second bin should be empty
    
    def test_evaluate_multiple_expectations(self, sample_expectations):
        """Test analysis evaluation with multiple expectations."""
        analysis = Analysis(sample_expectations)
        
        datasets = {
            "test_comp": {
                "energy": np.array([0.5, 1.5, 2.5])  # Events in different bins
            }
        }
        
        parameters = {
            "flux_norm": 1.5,
            "baseline_weight": 2.0
        }
        
        hist, hist_ssq = analysis.evaluate(datasets, parameters)
        
        # Should have results for both detectors
        assert "detector1" in hist
        assert "detector2" in hist
        assert "detector1" in hist_ssq
        assert "detector2" in hist_ssq
        
        # Check that histograms have correct shapes
        assert hist["detector1"].shape == (3,)  # 3 bins for detector1
        assert hist["detector2"].shape == (2,)  # 2 bins for detector2
    
    def test_evaluate_complex_model(self):
        """Test analysis with a more complex model."""
        # Create complex model with multiple factors
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
        model = Model.from_pairs("complex_model", [
            ("astro_weight", astro_component),
            ("atmo_weight", atmo_component)
        ])
        
        # Create binning and expectation
        bin_edges = [np.array([0, 1, 2, 3, 4])]
        binning = RectangularBinning(bin_edges, ["energy"])
        expectation = BinnedExpectation("obs", model, binning)
        
        analysis = Analysis({"obs": expectation})
        
        # Test with datasets for both components
        datasets = {
            "astro": {
                "log10_true_energy": np.array([4.0, 5.0, 6.0]),
                "energy": np.array([1.0, 2.0, 3.0])
            },
            "atmo": {
                "energy": np.array([0.5, 1.5, 2.5, 3.5])
            }
        }
        
        parameters = {
            "astro_norm": 1.0,
            "astro_index": -2.0,
            "atmo_norm": 0.5,
            "astro_weight": 1.0,
            "atmo_weight": 1.0
        }
        
        hist, hist_ssq = analysis.evaluate(datasets, parameters)
        
        assert "obs" in hist
        assert "obs" in hist_ssq
        assert hist["obs"].shape == (4,)  # 4 bins
        assert np.all(hist["obs"] >= 0)  # All bins should be non-negative
    
    def test_empty_datasets(self, sample_expectations):
        """Test analysis evaluation with empty datasets."""
        analysis = Analysis(sample_expectations)
        
        # Empty datasets
        datasets = {
            "test_comp": {
                "energy": np.array([])
            }
        }
        
        parameters = {
            "flux_norm": 1.0,
            "baseline_weight": 1.0
        }
        
        hist, hist_ssq = analysis.evaluate(datasets, parameters)
        
        # Should get zero histograms
        assert np.allclose(hist["detector1"], 0.0)
        assert np.allclose(hist["detector2"], 0.0)
        assert np.allclose(hist_ssq["detector1"], 0.0)
        assert np.allclose(hist_ssq["detector2"], 0.0)
    
    def test_mismatched_parameters(self, sample_expectations):
        """Test analysis evaluation with missing parameters."""
        analysis = Analysis(sample_expectations)
        
        datasets = {
            "test_comp": {
                "energy": np.array([0.5])
            }
        }
        
        # Missing required parameter
        parameters = {
            "flux_norm": 1.0
            # Missing "baseline_weight"
        }
        
        # Should raise an error for missing parameters
        with pytest.raises(KeyError):
            analysis.evaluate(datasets, parameters)
    
    def test_analysis_with_different_binnings(self):
        """Test analysis with expectations using different binning schemes."""
        # Create different binnings
        energy_binning = RectangularBinning(
            [np.array([0, 1, 2, 3])], 
            ["energy"]
        )
        
        zenith_binning = RectangularBinning(
            [np.array([0, 0.5, 1.0])], 
            ["cos_zenith"]
        )
        
        # Create models (simplified for different variables)
        norm_factor = FluxNorm(name="norm")
        component = ModelComponent("test_comp", [norm_factor])
        model = Model("test_model", [component], ["baseline_weight"])
        
        # Create expectations with different binnings
        energy_exp = BinnedExpectation("energy_detector", model, energy_binning)
        zenith_exp = BinnedExpectation("zenith_detector", model, zenith_binning)
        
        analysis = Analysis({
            "energy_detector": energy_exp,
            "zenith_detector": zenith_exp
        })
        
        # Test with appropriate datasets
        datasets = {
            "test_comp": {
                "energy": np.array([0.5, 1.5]),
                "cos_zenith": np.array([0.25, 0.75])
            }
        }
        
        parameters = {
            "flux_norm": 1.0,
            "baseline_weight": 1.0
        }
        
        hist, hist_ssq = analysis.evaluate(datasets, parameters)
        
        # Check that both detectors produce results
        assert "energy_detector" in hist
        assert "zenith_detector" in hist
        assert hist["energy_detector"].shape == (3,)  # 3 energy bins
        assert hist["zenith_detector"].shape == (2,)  # 2 zenith bins


class TestAnalysisIntegration:
    """Test integration of Analysis with other components."""
    
    def test_analysis_workflow(self):
        """Test complete analysis workflow from config-like setup to evaluation."""
        # Simulate configuration-based setup
        
        # Create factors
        powerlaw = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18
        )
        
        norm_factor = FluxNorm(name="norm")
        
        # Create components
        signal_component = ModelComponent("signal", [powerlaw, norm_factor])
        
        # Create model
        model = Model("analysis_model", [signal_component], ["signal_weight"])
        
        # Create binning
        bin_edges = [
            np.array([3, 4, 5, 6, 7]),  # log10(energy) bins
            np.array([-1, 0, 1])        # cos(zenith) bins
        ]
        binning = RectangularBinning(bin_edges, ["log10_energy", "cos_zenith"])
        
        # Create expectation
        expectation = BinnedExpectation("main_detector", model, binning)
        
        # Create analysis
        analysis = Analysis({"main_detector": expectation})
        
        # Test complete evaluation
        datasets = {
            "signal": {
                "log10_true_energy": np.array([3.5, 4.5, 5.5, 6.5]),
                "log10_energy": np.array([3.5, 4.5, 5.5, 6.5]),
                "cos_zenith": np.array([0.5, -0.5, 0.8, 0.2])
            }
        }
        
        parameters = {
            "flux_norm": 1.2,
            "spectral_index": -2.5,
            "signal_weight": 1.0
        }
        
        # Verify required variables and parameters
        assert analysis.required_variables.issuperset({
            "log10_true_energy", "log10_energy", "cos_zenith"
        })
        assert analysis.exposed_parameters.issuperset({
            "flux_norm", "spectral_index"
        })
        
        # Evaluate analysis
        hist, hist_ssq = analysis.evaluate(datasets, parameters)
        
        # Verify results
        assert "main_detector" in hist
        main_hist = hist["main_detector"]
        assert main_hist.shape == (8,)  # 4 energy bins × 2 zenith bins
        assert np.all(main_hist >= 0)
        assert np.sum(main_hist) > 0  # Should have some events
    
    def test_analysis_parameter_validation(self):
        """Test that analysis properly validates parameters."""
        # Create simple analysis
        bin_edges = [np.array([0, 1, 2])]
        binning = RectangularBinning(bin_edges, ["energy"])
        
        powerlaw = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18,
            param_mapping={
                "flux_norm": "norm",
                "spectral_index": "index"
            }
        )
        
        component = ModelComponent("test_comp", [powerlaw])
        model = Model("test_model", [component], ["weight"])
        
        expectation = BinnedExpectation("detector", model, binning)
        analysis = Analysis({"detector": expectation})
        
        # Check exposed parameters
        expected_params = {"norm", "index"}
        assert analysis.exposed_parameters == expected_params
        
        # Test with correct parameters
        datasets = {"test_comp": {"log10_true_energy": np.array([5.0]), "energy": np.array([1.0])}}
        correct_params = {"norm": 1.0, "index": -2.0, "weight": 1.0}
        
        hist, hist_ssq = analysis.evaluate(datasets, correct_params)
        assert "detector" in hist
        
        # Test with incorrect parameters (should fail)
        incorrect_params = {"flux_norm": 1.0, "spectral_index": -2.0, "weight": 1.0}
        
        with pytest.raises(KeyError):
            analysis.evaluate(datasets, incorrect_params)
