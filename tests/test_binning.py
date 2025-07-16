"""
Tests for pyForwardFolding.binning module.
"""
import numpy as np
import pytest

from pyForwardFolding.backend import backend
from pyForwardFolding.binning import AbstractBinning, RectangularBinning


class TestAbstractBinning:
    """Test the AbstractBinning base class."""

    def test_init_default(self):
        """Test default initialization of AbstractBinning."""
        binning = AbstractBinning()
        assert binning.bin_indices_dict == {}
        assert binning.mask_dict == {}

    def test_init_with_parameters(self):
        """Test initialization with bin indices and mask dictionaries."""
        bin_indices_dict = {"dataset1": [1, 2, 3]}
        mask_dict = {"dataset1": backend.array([True, False, True])}
        
        binning = AbstractBinning(bin_indices_dict, mask_dict)
        assert binning.bin_indices_dict == bin_indices_dict
        assert binning.mask_dict == mask_dict

    def test_required_variables_not_implemented(self):
        """Test that required_variables property raises NotImplementedError."""
        binning = AbstractBinning()
        with pytest.raises(NotImplementedError):
            _ = binning.required_variables

    def test_construct_from_rectangular_binning(self):
        """Test construct_from with RectangularBinning type."""
        config = {
            "type": "RectangularBinning",
            "bin_vars_edges": [("energy", "linear", [1.0, 10.0, 5])]
        }
        binning = AbstractBinning.construct_from(config)
        assert isinstance(binning, RectangularBinning)

    def test_construct_from_relaxed_binning(self):
        """Test construct_from with RelaxedBinning type."""
        config = {
            "type": "RelaxedBinning",
            "bin_variable": "energy",
            "bin_edges": [1.0, 10.0, 5],
            "slope": 0.1
        }
        # RelaxedBinning raises NotImplementedError
        with pytest.raises(NotImplementedError):
            AbstractBinning.construct_from(config)

    def test_construct_from_unknown_type(self):
        """Test construct_from with unknown binning type."""
        config = {
            "type": "UnknownBinning"
        }
        with pytest.raises(ValueError, match="Unknown binning type: UnknownBinning"):
            AbstractBinning.construct_from(config)

    def test_hist_dims_property(self):
        """Test hist_dims property calculation."""
        binning = AbstractBinning()
        # Mock bin_edges for testing
        binning.bin_edges = ([1, 2, 3, 4], [5, 6, 7])  # 3 bins, 2 bins
        expected_dims = (3, 2)
        assert binning.hist_dims == expected_dims

    def test_nbins_property(self):
        """Test nbins property calculation."""
        binning = AbstractBinning()
        # Mock bin_edges for testing
        binning.bin_edges = ([1, 2, 3, 4], [5, 6, 7])  # 3 bins, 2 bins
        expected_nbins = 3 * 2
        assert binning.nbins == expected_nbins

    def test_build_histogram_not_implemented(self):
        """Test that build_histogram raises NotImplementedError."""
        binning = AbstractBinning()
        with pytest.raises(NotImplementedError):
            binning.build_histogram(np.array([1, 2, 3]), (np.array([1, 2, 3]),))


class TestRectangularBinning:
    """Test RectangularBinning implementation."""

    def test_init(self):
        """Test RectangularBinning initialization."""
        bin_variables = ("energy", "angle")
        bin_edges = ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0])
        
        binning = RectangularBinning(bin_variables, bin_edges)
        assert binning.bin_variables == bin_variables
        assert len(binning.bin_edges) == 2

    def test_construct_from_linear_edges(self):
        """Test construction from config with linear edges."""
        config = {
            "bin_vars_edges": [
                ("energy", "linear", [1.0, 10.0, 5]),
                ("angle", "linear", [0.0, 1.0, 3])
            ]
        }
        binning = RectangularBinning.construct_from(config)
        assert binning.bin_variables == ["energy", "angle"]  # Returns list, not tuple
        assert len(binning.bin_edges) == 2
        assert len(binning.bin_edges[0]) == 5  # Linear spacing creates 5 points
        assert len(binning.bin_edges[1]) == 3  # Linear spacing creates 3 points

    def test_construct_from_array_edges(self):
        """Test construction from config with array edges."""
        config = {
            "bin_vars_edges": [
                ("energy", "array", [1.0, 2.0, 5.0, 10.0])
            ]
        }
        binning = RectangularBinning.construct_from(config)
        assert binning.bin_variables == ("energy",)
        assert len(binning.bin_edges) == 1
        assert len(binning.bin_edges[0]) == 4

    def test_construct_from_mixed_edges(self):
        """Test construction from config with mixed edge types."""
        config = {
            "bin_vars_edges": [
                ("energy", "linear", [1.0, 10.0, 5]),
                ("angle", "array", [0.0, 0.3, 0.7, 1.0])
            ]
        }
        binning = RectangularBinning.construct_from(config)
        assert binning.bin_variables == ("energy", "angle")
        assert len(binning.bin_edges) == 2

    def test_construct_from_empty_variables(self):
        """Test construction fails with empty variables."""
        config = {
            "bin_vars_edges": []
        }
        with pytest.raises(ValueError, match="At least one variable"):
            RectangularBinning.construct_from(config)

    def test_construct_from_unknown_bin_type(self):
        """Test construction fails with unknown bin type."""
        config = {
            "bin_vars_edges": [
                ("energy", "unknown_type", [1.0, 10.0, 5])
            ]
        }
        with pytest.raises(ValueError, match="Unknown binning type: unknown_type"):
            RectangularBinning.construct_from(config)

    def test_required_variables(self):
        """Test required_variables property."""
        bin_variables = ("energy", "angle")
        bin_edges = ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0])
        
        binning = RectangularBinning(bin_variables, bin_edges)
        assert binning.required_variables == ["energy", "angle"]

    def test_hist_dims_property(self):
        """Test hist_dims property for rectangular binning."""
        bin_variables = ("energy", "angle")
        bin_edges = ([1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0])  # 3 bins, 2 bins
        
        binning = RectangularBinning(bin_variables, bin_edges)
        assert binning.hist_dims == (3, 2)

    def test_nbins_property(self):
        """Test nbins property for rectangular binning."""
        bin_variables = ("energy", "angle") 
        bin_edges = ([1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0])  # 3 bins, 2 bins
        
        binning = RectangularBinning(bin_variables, bin_edges)
        assert binning.nbins == 6

    def test_calculate_bin_indices_single_variable(self):
        """Test calculate_bin_indices with single variable."""
        bin_variables = ("energy",)
        bin_edges = ([1.0, 2.0, 3.0, 4.0],)  # 3 bins
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        # Test data: values 1.5, 2.5, 3.5 should go to bins 0, 1, 2
        binning_variables = (backend.array([1.5, 2.5, 3.5]),)
        binning.calculate_bin_indices("test_dataset", binning_variables)
        
        # Check that bin indices were calculated
        assert "test_dataset" in binning.bin_indices_dict
        assert "test_dataset" in binning.mask_dict
        
        # Check bin indices - should be [0, 1, 2]
        expected_indices = [0, 1, 2]
        np.testing.assert_array_equal(binning.bin_indices_dict["test_dataset"][0], expected_indices)

    def test_calculate_bin_indices_multiple_variables(self):
        """Test calculate_bin_indices with multiple variables."""
        bin_variables = ("energy", "angle")
        bin_edges = ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0])  # 2 bins each
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        # Test data
        energy_data = backend.array([1.5, 2.5])
        angle_data = backend.array([0.25, 0.75])
        binning_variables = (energy_data, angle_data)
        
        binning.calculate_bin_indices("test_dataset", binning_variables)
        
        # Check that bin indices were calculated for both variables
        assert len(binning.bin_indices_dict["test_dataset"]) == 2
        
        # Energy: 1.5 -> bin 0, 2.5 -> bin 1
        # Angle: 0.25 -> bin 0, 0.75 -> bin 1
        expected_energy_indices = [0, 1]
        expected_angle_indices = [0, 1]
        
        np.testing.assert_array_equal(binning.bin_indices_dict["test_dataset"][0], expected_energy_indices)
        np.testing.assert_array_equal(binning.bin_indices_dict["test_dataset"][1], expected_angle_indices)

    def test_calculate_bin_indices_with_masking(self):
        """Test calculate_bin_indices with out-of-bounds values (masking)."""
        bin_variables = ("energy",)
        bin_edges = ([1.0, 2.0, 3.0],)  # 2 bins: [1-2), [2-3)
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        # Test data with out-of-bounds values
        binning_variables = (backend.array([0.5, 1.5, 2.5, 3.5]),)  # First and last are out of bounds
        binning.calculate_bin_indices("test_dataset", binning_variables)
        
        # Check mask - first and last should be True (masked)
        expected_mask = [True, False, False, True]
        np.testing.assert_array_equal(binning.mask_dict["test_dataset"], expected_mask)

    def test_calculate_bin_indices_mismatched_lengths(self):
        """Test calculate_bin_indices fails with mismatched variable lengths."""
        bin_variables = ("energy", "angle")
        bin_edges = ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0])
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        # Mismatched lengths
        energy_data = backend.array([1.5, 2.5])
        angle_data = backend.array([0.25])  # Different length
        binning_variables = (energy_data, angle_data)
        
        with pytest.raises(ValueError, match="All binning variables must have the same length"):
            binning.calculate_bin_indices("test_dataset", binning_variables)

    def test_build_histogram_1d(self):
        """Test build_histogram with 1D data."""
        bin_variables = ("energy",)
        bin_edges = ([1.0, 2.0, 3.0, 4.0],)  # 3 bins
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        # Test data and weights
        binning_variables = (backend.array([1.5, 2.5, 2.5, 3.5]),)
        weights = backend.array([1.0, 2.0, 3.0, 4.0])
        
        histogram = binning.build_histogram("test_dataset", weights, binning_variables)
        
        # Expected: bin 0 gets weight 1.0, bin 1 gets weights 2.0+3.0=5.0, bin 2 gets weight 4.0
        expected = np.array([1.0, 5.0, 4.0])
        np.testing.assert_array_equal(histogram, expected)

    def test_build_histogram_2d(self):
        """Test build_histogram with 2D data."""
        bin_variables = ("energy", "angle")
        bin_edges = ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0])  # 2x2 bins
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        # Test data: 4 events in different bins
        energy_data = backend.array([1.5, 1.5, 2.5, 2.5])
        angle_data = backend.array([0.25, 0.75, 0.25, 0.75])
        binning_variables = (energy_data, angle_data)
        weights = backend.array([1.0, 2.0, 3.0, 4.0])
        
        histogram = binning.build_histogram("test_dataset", weights, binning_variables)
        
        # Expected 2x2 histogram:
        # [(1.5, 0.25): 1.0, (1.5, 0.75): 2.0]
        # [(2.5, 0.25): 3.0, (2.5, 0.75): 4.0]
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(histogram, expected)

    def test_build_histogram_with_masking(self):
        """Test build_histogram with masked (out-of-bounds) values."""
        bin_variables = ("energy",)
        bin_edges = ([1.0, 2.0, 3.0],)  # 2 bins
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        # Test data with out-of-bounds values
        binning_variables = (backend.array([0.5, 1.5, 2.5, 3.5]),)  # First and last are out of bounds
        weights = backend.array([10.0, 1.0, 2.0, 20.0])  # Out-of-bounds weights should be ignored
        
        histogram = binning.build_histogram("test_dataset", weights, binning_variables)
        
        # Expected: only middle two values contribute
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_equal(histogram, expected)


class TestBinningIntegration:
    """Integration tests for binning functionality."""

    def test_multiple_datasets_same_binning(self):
        """Test that same binning can handle multiple datasets."""
        bin_variables = ("energy",)
        bin_edges = ([1.0, 2.0, 3.0],)
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        # First dataset
        binning_vars_1 = (backend.array([1.5, 2.5]),)
        weights_1 = backend.array([1.0, 2.0])
        hist_1 = binning.build_histogram("dataset1", weights_1, binning_vars_1)
        
        # Second dataset
        binning_vars_2 = (backend.array([1.5, 1.5]),)
        weights_2 = backend.array([3.0, 4.0])
        hist_2 = binning.build_histogram("dataset2", weights_2, binning_vars_2)
        
        # Check both datasets were processed correctly
        expected_1 = np.array([1.0, 2.0])
        expected_2 = np.array([7.0, 0.0])
        
        np.testing.assert_array_equal(hist_1, expected_1)
        np.testing.assert_array_equal(hist_2, expected_2)
        
        # Check that both datasets have their own indices and masks
        assert "dataset1" in binning.bin_indices_dict
        assert "dataset2" in binning.bin_indices_dict
        assert "dataset1" in binning.mask_dict
        assert "dataset2" in binning.mask_dict

    def test_reuse_calculated_indices(self):
        """Test that bin indices are reused when calculated multiple times."""
        bin_variables = ("energy",)
        bin_edges = ([1.0, 2.0, 3.0],)
        
        binning = RectangularBinning(bin_variables, bin_edges)
        
        binning_vars = (backend.array([1.5, 2.5]),)
        weights_1 = backend.array([1.0, 2.0])
        weights_2 = backend.array([10.0, 20.0])
        
        # Build histogram twice with same binning variables
        hist_1 = binning.build_histogram("dataset", weights_1, binning_vars)
        hist_2 = binning.build_histogram("dataset", weights_2, binning_vars)
        
        # Results should scale with weights
        expected_1 = np.array([1.0, 2.0])
        expected_2 = np.array([10.0, 20.0])
        
        np.testing.assert_array_equal(hist_1, expected_1)
        np.testing.assert_array_equal(hist_2, expected_2)


if __name__ == "__main__":
    pytest.main([__file__])
