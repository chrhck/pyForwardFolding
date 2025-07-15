"""Tests for the likelihood module."""

import numpy as np
import pytest

from pyForwardFolding.analysis import Analysis
from pyForwardFolding.binned_expectation import BinnedExpectation
from pyForwardFolding.binning import RectangularBinning
from pyForwardFolding.factor import FluxNorm
from pyForwardFolding.likelihood import (
    AbstractLikelihood,
    GaussianUnivariatePrior,
    PoissonLikelihood,
    SAYLikelihood,
)
from pyForwardFolding.model import Model
from pyForwardFolding.model_component import ModelComponent


class MockLikelihood(AbstractLikelihood):
    """Mock likelihood for testing abstract functionality."""
    
    def llh(self, observed_data, datasets, exposed_variables, empty_bins="skip"):
        return 42.0  # Mock constant likelihood


class TestAbstractLikelihood:
    """Test the abstract likelihood base class."""
    
    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for testing."""
        # Create a simple binning
        bin_edges = [np.array([0, 1, 2, 3])]
        binning = RectangularBinning(bin_edges, ["energy"])
        
        # Create a simple model
        norm_factor = FluxNorm(name="norm")
        component = ModelComponent("test_comp", [norm_factor])
        model = Model("test_model", [component], ["baseline_weight"])
        
        # Create binned expectation
        expectation = BinnedExpectation("test_exp", model, binning)
        
        # Create analysis
        return Analysis({"test_exp": expectation})
    
    def test_initialization(self, sample_analysis):
        """Test AbstractLikelihood initialization."""
        likelihood = MockLikelihood(sample_analysis)
        assert likelihood.analysis == sample_analysis
    
    def test_get_analysis(self, sample_analysis):
        """Test get_analysis method."""
        likelihood = MockLikelihood(sample_analysis)
        retrieved_analysis = likelihood.get_analysis()
        assert retrieved_analysis == sample_analysis
    
    def test_abstract_llh_not_implemented(self, sample_analysis):
        """Test that abstract llh method raises NotImplementedError."""
        likelihood = AbstractLikelihood(sample_analysis)
        
        with pytest.raises(NotImplementedError):
            likelihood.llh({}, {}, {})


class TestPoissonLikelihood:
    """Test the PoissonLikelihood class."""
    
    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for testing."""
        # Create a simple binning
        bin_edges = [np.array([0, 1, 2, 3])]
        binning = RectangularBinning(bin_edges, ["energy"])
        
        # Create a simple model
        norm_factor = FluxNorm(name="norm")
        component = ModelComponent("test_comp", [norm_factor])
        model = Model("test_model", [component], ["baseline_weight"])
        
        # Create binned expectation
        expectation = BinnedExpectation("test_exp", model, binning)
        
        # Create analysis
        return Analysis({"test_exp": expectation})
    
    def test_initialization(self, sample_analysis):
        """Test PoissonLikelihood initialization."""
        likelihood = PoissonLikelihood(sample_analysis)
        assert likelihood.analysis == sample_analysis
    
    def test_llh_perfect_fit(self, sample_analysis):
        """Test likelihood when prediction matches observation exactly."""
        likelihood = PoissonLikelihood(sample_analysis)
        
        # Mock datasets and parameters that will produce a known prediction
        datasets = {
            "test_comp": {
                "energy": np.array([0.5, 1.5, 2.5])  # 3 events in 3 bins
            }
        }
        
        parameters = {
            "flux_norm": 1.0,
            "baseline_weight": 1.0
        }
        
        # Observed data matching the prediction (1 event per bin)
        observed_data = {"test_exp": np.array([1.0, 1.0, 1.0])}
        
        # Calculate likelihood
        llh_value = likelihood.llh(observed_data, datasets, parameters)
        
        # For perfect Poisson fit, -2*log(L) should be minimal
        assert isinstance(llh_value, (float, np.floating))
        assert llh_value >= 0  # -2*log(L) is always non-negative
    
    def test_llh_zero_prediction(self, sample_analysis):
        """Test likelihood when prediction is zero."""
        likelihood = PoissonLikelihood(sample_analysis)
        
        # Empty datasets (no events)
        datasets = {"test_comp": {"energy": np.array([])}}
        
        parameters = {
            "flux_norm": 1.0,
            "baseline_weight": 1.0
        }
        
        # Observed data with zero counts
        observed_data = {"test_exp": np.array([0.0, 0.0, 0.0])}
        
        llh_value = likelihood.llh(observed_data, datasets, parameters)
        
        # Should handle zero prediction gracefully
        assert np.isfinite(llh_value)
    
    def test_llh_empty_bins_skip(self, sample_analysis):
        """Test likelihood with empty bins and skip option."""
        likelihood = PoissonLikelihood(sample_analysis)
        
        datasets = {"test_comp": {"energy": np.array([0.5])}}  # Only one event
        parameters = {"flux_norm": 1.0, "baseline_weight": 1.0}
        
        # Observed data with some zero bins
        observed_data = {"test_exp": np.array([1.0, 0.0, 0.0])}
        
        llh_value = likelihood.llh(observed_data, datasets, parameters, empty_bins="skip")
        
        assert np.isfinite(llh_value)
    
    def test_llh_empty_bins_error(self, sample_analysis):
        """Test likelihood with empty bins and error option."""
        likelihood = PoissonLikelihood(sample_analysis)
        
        datasets = {"test_comp": {"energy": np.array([])}}  # No events
        parameters = {"flux_norm": 1.0, "baseline_weight": 1.0}
        
        # Observed data with non-zero counts but zero prediction
        observed_data = {"test_exp": np.array([1.0, 0.0, 0.0])}
        
        with pytest.raises(ValueError, match="Zero prediction with non-zero observation"):
            likelihood.llh(observed_data, datasets, parameters, empty_bins="error")


class TestSAYLikelihood:
    """Test the SAYLikelihood (Simplified ANTARES-IceCube) class."""
    
    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for testing."""
        # Create a simple binning
        bin_edges = [np.array([0, 1, 2, 3])]
        binning = RectangularBinning(bin_edges, ["energy"])
        
        # Create a simple model
        norm_factor = FluxNorm(name="norm")
        component = ModelComponent("test_comp", [norm_factor])
        model = Model("test_model", [component], ["baseline_weight"])
        
        # Create binned expectation
        expectation = BinnedExpectation("test_exp", model, binning)
        
        # Create analysis
        return Analysis({"test_exp": expectation})
    
    def test_initialization(self, sample_analysis):
        """Test SAYLikelihood initialization."""
        likelihood = SAYLikelihood(sample_analysis)
        assert likelihood.analysis == sample_analysis
    
    def test_llh_basic(self, sample_analysis):
        """Test basic SAY likelihood calculation."""
        likelihood = SAYLikelihood(sample_analysis)
        
        # Mock datasets
        datasets = {
            "test_comp": {
                "energy": np.array([0.5, 1.5, 2.5])
            }
        }
        
        parameters = {
            "flux_norm": 1.0,
            "baseline_weight": 1.0
        }
        
        # Observed data
        observed_data = {"test_exp": np.array([1.0, 2.0, 1.0])}
        
        llh_value = likelihood.llh(observed_data, datasets, parameters)
        
        # SAY likelihood should be finite and non-negative
        assert np.isfinite(llh_value)
        assert llh_value >= 0
    
    def test_llh_with_uncertainties(self, sample_analysis):
        """Test SAY likelihood with uncertainties."""
        likelihood = SAYLikelihood(sample_analysis)
        
        datasets = {
            "test_comp": {
                "energy": np.array([0.5, 1.5, 2.5])
            }
        }
        
        parameters = {
            "flux_norm": 1.0,
            "baseline_weight": 1.0
        }
        
        # Observed data with statistical uncertainties
        observed_data = {"test_exp": np.array([1.0, 2.0, 1.0])}
        
        llh_value = likelihood.llh(observed_data, datasets, parameters)
        
        assert np.isfinite(llh_value)


class TestGaussianUnivariatePrior:
    """Test the GaussianUnivariatePrior class."""
    
    def test_initialization(self):
        """Test GaussianUnivariatePrior initialization."""
        prior = GaussianUnivariatePrior("test_param", mean=1.0, sigma=0.1)
        
        assert prior.parameter_name == "test_param"
        assert prior.mean == 1.0
        assert prior.sigma == 0.1
    
    def test_evaluate_at_mean(self):
        """Test prior evaluation at the mean value."""
        prior = GaussianUnivariatePrior("test_param", mean=1.0, sigma=0.1)
        
        # At the mean, the prior contribution should be zero
        result = prior.evaluate({"test_param": 1.0})
        assert abs(result) < 1e-10  # Should be very close to zero
    
    def test_evaluate_away_from_mean(self):
        """Test prior evaluation away from the mean."""
        prior = GaussianUnivariatePrior("test_param", mean=1.0, sigma=0.1)
        
        # One sigma away from mean
        result = prior.evaluate({"test_param": 1.1})
        expected = 0.5 * ((1.1 - 1.0) / 0.1) ** 2  # 0.5 * chi^2
        assert abs(result - expected) < 1e-10
    
    def test_evaluate_multiple_sigmas(self):
        """Test prior evaluation at multiple sigma values."""
        prior = GaussianUnivariatePrior("test_param", mean=0.0, sigma=1.0)
        
        # Test at different sigma values
        for n_sigma in [1, 2, 3]:
            result = prior.evaluate({"test_param": n_sigma})
            expected = 0.5 * n_sigma ** 2
            assert abs(result - expected) < 1e-10
    
    def test_evaluate_missing_parameter(self):
        """Test prior evaluation when parameter is missing."""
        prior = GaussianUnivariatePrior("test_param", mean=1.0, sigma=0.1)
        
        # Parameter not in dictionary
        with pytest.raises(KeyError):
            prior.evaluate({"other_param": 1.0})


class TestLikelihoodIntegration:
    """Test integration between likelihood and other components."""
    
    def test_likelihood_with_complex_model(self):
        """Test likelihood with a more complex model."""
        # Create a more complex analysis setup
        bin_edges = [np.array([0, 1, 2, 3, 4])]
        binning = RectangularBinning(bin_edges, ["energy"])
        
        # Multiple factors and components
        from pyForwardFolding.factor import PowerLawFlux
        
        powerlaw = PowerLawFlux(
            name="powerlaw",
            pivot_energy=1e5,
            baseline_norm=1e-18,
            param_mapping={
                "flux_norm": "astro_norm",
                "spectral_index": "astro_index"
            }
        )
        
        norm_factor = FluxNorm(
            name="atmo_norm",
            param_mapping={"flux_norm": "atmo_norm"}
        )
        
        astro_component = ModelComponent("astro", [powerlaw])
        atmo_component = ModelComponent("atmo", [norm_factor])
        
        model = Model.from_pairs("complex_model", [
            ("astro_weight", astro_component),
            ("atmo_weight", atmo_component)
        ])
        
        expectation = BinnedExpectation("obs", model, binning)
        analysis = Analysis({"obs": expectation})
        
        # Test with Poisson likelihood
        likelihood = PoissonLikelihood(analysis)
        
        datasets = {
            "astro": {"log10_true_energy": np.array([4.0, 5.0, 6.0])},
            "atmo": {"energy": np.array([0.5, 1.5, 2.5, 3.5])}
        }
        
        parameters = {
            "astro_norm": 1.0,
            "astro_index": -2.0,
            "atmo_norm": 0.5,
            "astro_weight": 1.0,
            "atmo_weight": 1.0
        }
        
        observed_data = {"obs": np.array([2.0, 1.0, 3.0, 1.0])}
        
        llh_value = likelihood.llh(observed_data, datasets, parameters)
        
        assert np.isfinite(llh_value)
        assert llh_value >= 0
    
    def test_likelihood_with_priors(self):
        """Test likelihood combined with priors."""
        # Simple analysis setup
        bin_edges = [np.array([0, 1, 2])]
        binning = RectangularBinning(bin_edges, ["energy"])
        
        norm_factor = FluxNorm(name="norm")
        component = ModelComponent("test_comp", [norm_factor])
        model = Model("test_model", [component], ["baseline_weight"])
        
        expectation = BinnedExpectation("test_exp", model, binning)
        analysis = Analysis({"test_exp": expectation})
        
        likelihood = PoissonLikelihood(analysis)
        
        # Create priors
        norm_prior = GaussianUnivariatePrior("flux_norm", mean=1.0, sigma=0.1)
        weight_prior = GaussianUnivariatePrior("baseline_weight", mean=1.0, sigma=0.2)
        
        datasets = {"test_comp": {"energy": np.array([0.5])}}
        parameters = {"flux_norm": 1.05, "baseline_weight": 0.95}
        observed_data = {"test_exp": np.array([1.0, 0.0])}
        
        # Calculate likelihood and priors
        llh_value = likelihood.llh(observed_data, datasets, parameters)
        prior_value = norm_prior.evaluate(parameters) + weight_prior.evaluate(parameters)
        
        total_objective = llh_value + prior_value
        
        assert np.isfinite(total_objective)
        assert total_objective > llh_value  # Priors should add positive contribution
