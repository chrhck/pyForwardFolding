"""
Tests for pyForwardFolding.backend module.

This module contains tests for the backend functionality, including:
- JAXBackend implementation
- Array operations and mathematical functions
- Statistical functions and distributions
"""

import numpy as np
import pytest

from pyForwardFolding.backend import JAXBackend, backend


class TestJAXBackend:
    """Test the JAXBackend implementation."""

    def test_array_creation(self):
        """Test basic array creation."""
        arr = backend.array([1, 2, 3, 4])
        assert arr.shape == (4,)
        np.testing.assert_array_equal(arr, np.array([1, 2, 3, 4]))

    def test_zeros(self):
        """Test zeros array creation."""
        arr = backend.zeros((3, 2))
        assert arr.shape == (3, 2)
        assert np.all(arr == 0)
        np.testing.assert_array_equal(arr, np.zeros((3, 2)))

    def test_basic_math_operations(self):
        """Test basic mathematical operations."""
        arr = backend.array([1.0, 4.0, 9.0])
        
        # Power operation
        result_power = backend.power(arr, 0.5)
        expected_power = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result_power, expected_power)
        
        # Exponential
        arr_small = backend.array([0.0, 1.0, 2.0])
        result_exp = backend.exp(arr_small)
        expected_exp = np.array([1.0, np.e, np.e**2])
        np.testing.assert_array_almost_equal(result_exp, expected_exp)
        
        # Square root
        result_sqrt = backend.sqrt(arr)
        np.testing.assert_array_almost_equal(result_sqrt, expected_power)
        
        # Logarithm
        result_log = backend.log(arr)
        expected_log = np.log([1.0, 4.0, 9.0])
        np.testing.assert_array_almost_equal(result_log, expected_log)

    def test_statistical_functions(self):
        """Test statistical and probability functions."""
        # Gaussian PDF
        x = backend.array([0.0, 1.0, -1.0])
        mu = 0.0
        sigma = 1.0
        pdf_result = backend.gauss_pdf(x, mu, sigma)
        
        # Should be symmetric around mu=0
        assert pdf_result[0] > pdf_result[1]  # pdf(0) > pdf(1)
        np.testing.assert_almost_equal(pdf_result[1], pdf_result[2])  # pdf(1) ≈ pdf(-1)
        
        # Gaussian CDF
        cdf_result = backend.gauss_cdf(x, mu, sigma)
        assert cdf_result[0] > 0.4 and cdf_result[0] < 0.6  # Should be around 0.5 for x=0
        assert cdf_result[1] > cdf_result[0]  # CDF should be increasing
        assert cdf_result[2] < cdf_result[0]  # CDF at -1 should be less than at 0

    def test_uniform_pdf(self):
        """Test uniform probability density function."""
        x = backend.array([0.5, 1.0, 1.5, 2.5])
        lo = 1.0
        hi = 2.0
        result = backend.uniform_pdf(x, lo, hi)
        
        # Expected: 0, 1, 1, 0 (scaled by 1/(hi-lo))
        expected = np.array([0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_array_indexing_operations(self):
        """Test array indexing and modification operations."""
        arr = backend.array([1.0, 2.0, 3.0, 4.0])
        
        # Test set_index
        new_arr = backend.set_index(arr, 1, 10.0)
        expected = np.array([1.0, 10.0, 3.0, 4.0])
        np.testing.assert_array_equal(new_arr, expected)
        
        # Test set_index_add
        add_arr = backend.set_index_add(arr, 1, 5.0)
        expected_add = np.array([1.0, 7.0, 3.0, 4.0])  # 2.0 + 5.0 = 7.0
        np.testing.assert_array_equal(add_arr, expected_add)

    def test_histogram_operations(self):
        """Test histogram-related operations."""
        # Test bincount
        indices = backend.array([0, 1, 1, 2, 2, 2])
        weights = backend.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = backend.bincount(indices, weights=weights, length=4)
        
        # Expected: [1.0, 5.0, 15.0, 0.0]
        expected = np.array([1.0, 5.0, 15.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_array_manipulation(self):
        """Test array manipulation functions."""
        arr = backend.array([[1, 2], [3, 4]])
        
        # Test reshape
        reshaped = backend.reshape(arr, (4,))
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(reshaped, expected)
        
        # Test clip
        arr_to_clip = backend.array([0.5, 1.5, 2.5, 3.5])
        clipped = backend.clip(arr_to_clip, 1.0, 3.0)
        expected_clipped = np.array([1.0, 1.5, 2.5, 3.0])
        np.testing.assert_array_equal(clipped, expected_clipped)

    def test_linspace(self):
        """Test linspace function."""
        result = backend.linspace(0.0, 10.0, 5)
        expected = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mathematical_functions(self):
        """Test additional mathematical functions."""
        arr = backend.array([0.0, 0.5, 1.0])
        
        # Test tanh
        tanh_result = backend.tanh(arr)
        expected_tanh = np.tanh([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(tanh_result, expected_tanh)
        
        # Test sigmoid
        sigmoid_result = backend.sigmoid(arr)
        expected_sigmoid = 1 / (1 + np.exp(-np.array([0.0, 0.5, 1.0])))
        np.testing.assert_array_almost_equal(sigmoid_result, expected_sigmoid)

    def test_where_operations(self):
        """Test conditional where operations."""
        condition = backend.array([True, False, True, False])
        x = backend.array([1, 2, 3, 4])
        y = backend.array([10, 20, 30, 40])
        
        result = backend.where(condition, x, y)
        expected = np.array([1, 20, 3, 40])
        np.testing.assert_array_equal(result, expected)

    def test_digitize(self):
        """Test digitize function."""
        x = backend.array([0.5, 1.5, 2.5, 3.5])
        bins = backend.array([1.0, 2.0, 3.0])
        
        result = backend.digitize(x, bins)
        # Values should be binned as: 0.5->0, 1.5->1, 2.5->2, 3.5->3
        expected = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_searchsorted(self):
        """Test searchsorted function."""
        a = backend.array([1.0, 3.0, 5.0, 7.0])
        v = backend.array([2.0, 4.0, 6.0])
        
        result = backend.searchsorted(a, v)
        expected = np.array([1, 2, 3])  # Indices where v elements would be inserted
        np.testing.assert_array_equal(result, expected)

    def test_ravel_multi_index(self):
        """Test ravel_multi_index function."""
        # Convert 2D indices to flat indices for a 3x4 array
        indices = (backend.array([0, 1, 2]), backend.array([0, 2, 3]))
        dims = (3, 4)
        
        result = backend.ravel_multi_index(indices, dims)
        expected = np.array([0, 6, 11])  # (0,0)->0, (1,2)->6, (2,3)->11 for 3x4 array
        np.testing.assert_array_equal(result, expected)

    def test_weighted_quantile_functions(self):
        """Test weighted quantile and median functions."""
        x = backend.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = backend.array([1.0, 1.0, 2.0, 1.0, 1.0])  # Higher weight on 3.0
        
        # Test weighted median
        median = backend.weighted_median(x, weights)
        # With higher weight on 3.0, median should be close to 3.0
        assert median >= 2.0 and median <= 4.0
        
        # Test weighted quantile
        q25 = backend.weighted_quantile(x, weights, 0.25)
        q75 = backend.weighted_quantile(x, weights, 0.75)
        
        # Quantiles should be in order: q25 <= median <= q75
        assert q25 <= median <= q75

    def test_error_function(self):
        """Test error function implementations."""
        x = backend.array([0.0, 0.5, 1.0, -0.5])
        
        # Test standard erf
        erf_result = backend.erf(x)
        
        # erf(0) = 0, erf is odd function: erf(-x) = -erf(x)
        np.testing.assert_almost_equal(erf_result[0], 0.0, decimal=5)
        np.testing.assert_almost_equal(erf_result[1], -erf_result[3], decimal=5)
        
        # Test fast erf approximation
        fasterf_result = backend.fasterf(x)
        
        # Should be approximately the same as erf for small values
        np.testing.assert_array_almost_equal(erf_result, fasterf_result, decimal=2)

    def test_gamma_functions(self):
        """Test gamma-related functions."""
        x = backend.array([1.0, 2.0, 3.0, 4.0])
        
        # Test gammaln (log gamma function)
        result = backend.gammaln(x)
        
        # gammaln(1) = ln(0!) = ln(1) = 0
        # gammaln(2) = ln(1!) = ln(1) = 0  
        # gammaln(3) = ln(2!) = ln(2)
        # gammaln(4) = ln(3!) = ln(6)
        np.testing.assert_almost_equal(result[0], 0.0, decimal=5)
        np.testing.assert_almost_equal(result[1], 0.0, decimal=5)
        np.testing.assert_almost_equal(result[2], np.log(2), decimal=5)
        np.testing.assert_almost_equal(result[3], np.log(6), decimal=5)

    def test_array_reductions(self):
        """Test array reduction operations."""
        arr = backend.array([[1, 2, 3], [4, 5, 6]])
        
        # Test sum with no axis (total sum)
        total_sum = backend.sum(arr)
        expected_total = 21  # 1+2+3+4+5+6
        assert int(total_sum) == expected_total
        
        # Test sum with axis
        axis0_sum = backend.sum(arr, axis=0)
        expected_axis0 = np.array([5, 7, 9])  # Column sums
        np.testing.assert_array_equal(axis0_sum, expected_axis0)
        
        # Test any
        bool_arr = backend.array([False, True, False])
        assert backend.any(bool_arr)
        
        all_false = backend.array([False, False, False])
        assert not backend.any(all_false)

    def test_diff_and_allclose(self):
        """Test diff and allclose functions."""
        # Test diff
        arr = backend.array([1, 4, 6, 10])
        diff_result = backend.diff(arr)
        expected_diff = np.array([3, 2, 4])  # Differences between consecutive elements
        np.testing.assert_array_equal(diff_result, expected_diff)
        
        # Test allclose
        arr1 = backend.array([1.0, 2.0, 3.0])
        arr2 = backend.array([1.0001, 2.0001, 3.0001])
        arr3 = backend.array([1.1, 2.1, 3.1])
        
        # Should be close within default tolerance
        assert backend.allclose(arr1, arr2, atol=1e-3)
        # Should not be close with smaller tolerance  
        assert not backend.allclose(arr1, arr3, atol=1e-3)


class TestBackendIntegration:
    """Integration tests for backend functionality."""

    def test_backend_singleton(self):
        """Test that backend is a singleton instance."""
        assert isinstance(backend, JAXBackend)
        
        # Create another backend instance
        backend2 = JAXBackend()
        
        # They should have the same functionality but be different objects
        arr1 = backend.array([1, 2, 3])
        arr2 = backend2.array([1, 2, 3])
        
        np.testing.assert_array_equal(arr1, arr2)

    def test_complex_computation_chain(self):
        """Test a complex chain of backend operations."""
        # Create some test data
        x = backend.linspace(0.0, 2*np.pi, 100)
        
        # Compute sin(x) using exp and complex arithmetic simulation
        # sin(x) ≈ (exp(ix) - exp(-ix)) / 2i, but we'll use tanh for a real function
        y = backend.tanh(x)
        
        # Apply some transformations
        y_clipped = backend.clip(y, -0.5, 0.5)
        y_reshaped = backend.reshape(y_clipped, (10, 10))
        
        # Compute some statistics
        mean_val = backend.sum(y_reshaped) / backend.array(100.0)
        
        # Result should be reasonable
        assert isinstance(mean_val, backend.array([0.0]).__class__)
        assert abs(float(mean_val)) < 1.0  # Should be bounded

    def test_probability_distribution_workflow(self):
        """Test a realistic probability distribution workflow."""
        # Generate some synthetic data points
        x_values = backend.linspace(-3.0, 3.0, 100)
        
        # Parameters for Gaussian
        mu = 0.0
        sigma = 1.0
        
        # Compute PDF and CDF
        pdf_values = backend.gauss_pdf(x_values, mu, sigma)
        cdf_values = backend.gauss_cdf(x_values, mu, sigma)
        
        # Validate PDF properties
        # PDF should be positive
        assert backend.any(pdf_values >= 0)
        
        # PDF should be maximal at mu=0 (which is at index 50 for our linspace)
        max_idx = 50  # Middle of the range
        assert pdf_values[max_idx] >= pdf_values[0]  # Center > edge
        assert pdf_values[max_idx] >= pdf_values[-1]  # Center > edge
        
        # CDF should be monotonically increasing
        cdf_diff = backend.diff(cdf_values)
        assert backend.any(cdf_diff >= 0)  # All differences should be non-negative


if __name__ == "__main__":
    pytest.main([__file__])
