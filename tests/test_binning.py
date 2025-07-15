"""Tests for the binning module."""

import numpy as np
import pytest

from pyForwardFolding.binning import AbstractBinning, CustomBinning, RectangularBinning


class MockBinning(AbstractBinning):
    """Mock binning for testing abstract functionality."""
    
    def __init__(self):
        super().__init__()
        self._required_variables = ["var1", "var2"]
        self._hist_dims = (10, 5)
    
    @property
    def required_variables(self):
        return self._required_variables
    
    @property
    def hist_dims(self):
        return self._hist_dims
    
    def build_histogram(self, weights, binning_variables):
        # Simple mock implementation
        return np.ones(self.nbins)


class TestAbstractBinning:
    """Test the abstract binning base class."""
    
    def test_initialization(self):
        """Test AbstractBinning initialization."""
        binning = MockBinning()
        assert binning.bin_indices_dict == {}
        assert binning.mask_dict == {}
        assert binning.required_variables == ["var1", "var2"]
        assert binning.hist_dims == (10, 5)
        assert binning.nbins == 50  # 10 * 5
    
    def test_initialization_with_dicts(self):
        """Test initialization with custom dictionaries."""
        bin_indices = {"test": np.array([1, 2, 3])}
        mask = {"test": np.array([True, False, True])}
        
        binning = AbstractBinning(bin_indices_dict=bin_indices, mask_dict=mask)
        assert binning.bin_indices_dict == bin_indices
        assert binning.mask_dict == mask
    
    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        binning = AbstractBinning()
        
        with pytest.raises(NotImplementedError):
            _ = binning.required_variables
        
        with pytest.raises(NotImplementedError):
            binning.construct_from({})
        
        with pytest.raises(NotImplementedError):
            _ = binning.hist_dims
        
        with pytest.raises(NotImplementedError):
            binning.build_histogram(np.array([1, 2, 3]), (np.array([1, 2, 3]),))


class TestRectangularBinning:
    """Test the RectangularBinning class."""
    
    def test_initialization_1d(self):
        """Test 1D rectangular binning initialization."""
        bin_edges = [np.array([0, 1, 2, 3, 4])]
        variable_names = ["energy"]
        
        binning = RectangularBinning(bin_edges, variable_names)
        
        assert binning.bin_edges == bin_edges
        assert binning.variable_names == variable_names
        assert binning.required_variables == ["energy"]
        assert binning.hist_dims == (4,)  # 5 edges = 4 bins
        assert binning.nbins == 4
    
    def test_initialization_2d(self):
        """Test 2D rectangular binning initialization."""
        bin_edges = [
            np.array([0, 1, 2, 3]),  # 3 bins
            np.array([0, 0.5, 1.0])  # 2 bins
        ]
        variable_names = ["energy", "coszen"]
        
        binning = RectangularBinning(bin_edges, variable_names)
        
        assert binning.hist_dims == (3, 2)
        assert binning.nbins == 6  # 3 * 2
        assert binning.required_variables == ["energy", "coszen"]
    
    def test_build_histogram_1d(self):
        """Test building 1D histogram."""
        bin_edges = [np.array([0, 1, 2, 3])]
        variable_names = ["energy"]
        binning = RectangularBinning(bin_edges, variable_names)
        
        # Events at energies 0.5, 1.5, 2.5, 1.2
        energy_values = np.array([0.5, 1.5, 2.5, 1.2])
        weights = np.array([1.0, 2.0, 3.0, 1.5])
        
        histogram = binning.build_histogram(weights, (energy_values,))
        
        # Expected: bin 0 gets weight 1.0, bin 1 gets weights 2.0+1.5=3.5, bin 2 gets weight 3.0
        expected = np.array([1.0, 3.5, 3.0])
        np.testing.assert_array_almost_equal(histogram, expected)
    
    def test_build_histogram_2d(self):
        """Test building 2D histogram."""
        bin_edges = [
            np.array([0, 1, 2]),  # 2 energy bins
            np.array([0, 0.5, 1])  # 2 coszen bins
        ]
        variable_names = ["energy", "coszen"]
        binning = RectangularBinning(bin_edges, variable_names)
        
        # Two events: (0.5, 0.25) and (1.5, 0.75)
        energy_values = np.array([0.5, 1.5])
        coszen_values = np.array([0.25, 0.75])
        weights = np.array([1.0, 2.0])
        
        histogram = binning.build_histogram(weights, (energy_values, coszen_values))
        
        # Should have shape (2, 2) flattened to (4,)
        # Event 1 goes to bin (0, 0), event 2 goes to bin (1, 1)
        assert histogram.shape == (4,)
        assert histogram[0] == 1.0  # bin (0, 0)
        assert histogram[3] == 2.0  # bin (1, 1)
        assert histogram[1] == 0.0  # bin (0, 1)
        assert histogram[2] == 0.0  # bin (1, 0)
    
    def test_construct_from(self):
        """Test constructing RectangularBinning from config."""
        config = {
            "type": "RectangularBinning",
            "bin_edges": [
                [0, 1, 2, 3],
                [0, 0.5, 1.0]
            ],
            "variable_names": ["energy", "coszen"]
        }
        
        binning = RectangularBinning.construct_from(config)
        
        assert binning.variable_names == ["energy", "coszen"]
        assert len(binning.bin_edges) == 2
        np.testing.assert_array_equal(binning.bin_edges[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(binning.bin_edges[1], [0, 0.5, 1.0])
    
    def test_empty_bins(self):
        """Test histogram with events outside bin ranges."""
        bin_edges = [np.array([1, 2, 3])]
        variable_names = ["energy"]
        binning = RectangularBinning(bin_edges, variable_names)
        
        # Events outside the bin range
        energy_values = np.array([0.5, 3.5])  # Below and above bin range
        weights = np.array([1.0, 2.0])
        
        histogram = binning.build_histogram(weights, (energy_values,))
        
        # Both events should be outside bins, so histogram should be all zeros
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(histogram, expected)


class TestCustomBinning:
    """Test the CustomBinning class."""
    
    def test_initialization(self):
        """Test CustomBinning initialization."""
        bin_indices = np.array([0, 1, 0, 2, 1])
        variable_names = ["energy"]
        nbins = 3
        
        binning = CustomBinning(bin_indices, variable_names, nbins)
        
        assert binning.variable_names == variable_names
        assert binning.nbins == nbins
        assert binning.required_variables == ["energy"]
        assert binning.hist_dims == (3,)
        np.testing.assert_array_equal(binning.bin_indices, bin_indices)
    
    def test_build_histogram(self):
        """Test building histogram with custom binning."""
        # Define custom bin assignments
        bin_indices = np.array([0, 1, 0, 2, 1])  # 5 events assigned to bins 0, 1, 0, 2, 1
        variable_names = ["event_id"]
        nbins = 3
        
        binning = CustomBinning(bin_indices, variable_names, nbins)
        
        # Weights for the 5 events
        weights = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
        # Variable values (not actually used in custom binning, but required for interface)
        variable_values = (np.array([0, 1, 2, 3, 4]),)
        
        histogram = binning.build_histogram(weights, variable_values)
        
        # Bin 0: events 0 and 2 -> weights 1.0 + 1.5 = 2.5
        # Bin 1: events 1 and 4 -> weights 2.0 + 2.5 = 4.5
        # Bin 2: event 3 -> weight 3.0
        expected = np.array([2.5, 4.5, 3.0])
        np.testing.assert_array_almost_equal(histogram, expected)
    
    def test_construct_from(self):
        """Test constructing CustomBinning from config."""
        config = {
            "type": "CustomBinning",
            "bin_indices": [0, 1, 0, 2, 1],
            "variable_names": ["energy"],
            "nbins": 3
        }
        
        binning = CustomBinning.construct_from(config)
        
        assert binning.variable_names == ["energy"]
        assert binning.nbins == 3
        np.testing.assert_array_equal(binning.bin_indices, [0, 1, 0, 2, 1])
    
    def test_mismatched_lengths(self):
        """Test that mismatched bin_indices and weights raise appropriate error."""
        bin_indices = np.array([0, 1, 2])
        variable_names = ["energy"]
        nbins = 3
        
        binning = CustomBinning(bin_indices, variable_names, nbins)
        
        # Provide weights with different length
        weights = np.array([1.0, 2.0])  # Only 2 weights for 3 bin indices
        variable_values = (np.array([0, 1]),)  # Only 2 variable values
        
        # This should work as long as weights and variable_values have same length
        histogram = binning.build_histogram(weights, variable_values)
        
        # Only first 2 events will be binned
        expected = np.array([1.0, 2.0, 0.0])
        np.testing.assert_array_almost_equal(histogram, expected)


class TestBinningIntegration:
    """Test integration between different binning classes."""
    
    def test_rectangular_vs_custom_consistency(self):
        """Test that RectangularBinning and CustomBinning give consistent results."""
        # Create a simple case where we can compare results
        
        # RectangularBinning setup
        bin_edges = [np.array([0, 1, 2, 3])]
        variable_names = ["energy"]
        rect_binning = RectangularBinning(bin_edges, variable_names)
        
        # Events that clearly fall into specific bins
        energy_values = np.array([0.5, 1.5, 2.5])  # Should go to bins 0, 1, 2
        weights = np.array([1.0, 2.0, 3.0])
        
        rect_histogram = rect_binning.build_histogram(weights, (energy_values,))
        
        # CustomBinning setup with equivalent bin assignments
        bin_indices = np.array([0, 1, 2])  # Manual assignment to same bins
        custom_binning = CustomBinning(bin_indices, variable_names, 3)
        
        custom_histogram = custom_binning.build_histogram(weights, (energy_values,))
        
        # Results should be identical
        np.testing.assert_array_almost_equal(rect_histogram, custom_histogram)
    
    def test_binning_with_zero_weights(self):
        """Test binning behavior with zero weights."""
        bin_edges = [np.array([0, 1, 2, 3])]
        variable_names = ["energy"]
        binning = RectangularBinning(bin_edges, variable_names)
        
        energy_values = np.array([0.5, 1.5, 2.5])
        weights = np.array([0.0, 0.0, 0.0])  # All zero weights
        
        histogram = binning.build_histogram(weights, (energy_values,))
        
        # All bins should be zero
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(histogram, expected)
    
    def test_binning_with_negative_weights(self):
        """Test binning behavior with negative weights."""
        bin_edges = [np.array([0, 1, 2])]
        variable_names = ["energy"]
        binning = RectangularBinning(bin_edges, variable_names)
        
        energy_values = np.array([0.5, 1.5])
        weights = np.array([1.0, -0.5])  # Mixed positive/negative
        
        histogram = binning.build_histogram(weights, (energy_values,))
        
        # Should handle negative weights correctly
        expected = np.array([1.0, -0.5])
        np.testing.assert_array_almost_equal(histogram, expected)
