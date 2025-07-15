"""Enhanced tests for backend numerical operations and edge cases."""

import jax.numpy as jnp
import numpy as np
import pytest

from pyForwardFolding.backend import JAXBackend


class TestJAXBackendNumericalStability:
    """Test numerical stability and edge cases for JAX backend."""
    
    @pytest.fixture
    def backend(self):
        """Provide a JAX backend instance."""
        return JAXBackend()
    
    def test_array_operations_with_nans(self, backend):
        """Test backend operations with NaN values."""
        arr_with_nan = backend.array([1.0, np.nan, 3.0])
        
        # Test that NaN propagates correctly
        result = backend.multiply(arr_with_nan, 2.0)
        assert np.isnan(result[1])
        assert result[0] == 2.0
        assert result[2] == 6.0
    
    def test_array_operations_with_inf(self, backend):
        """Test backend operations with infinity values."""
        arr_with_inf = backend.array([1.0, np.inf, -np.inf])
        
        # Test arithmetic with infinity
        result = backend.multiply(arr_with_inf, 2.0)
        assert result[0] == 2.0
        assert np.isinf(result[1]) and result[1] > 0
        assert np.isinf(result[2]) and result[2] < 0
    
    def test_division_by_zero_handling(self, backend):
        """Test division by zero behavior."""
        numerator = backend.array([1.0, 2.0, 3.0])
        denominator = backend.array([1.0, 0.0, 2.0])
        
        # JAX should handle division by zero according to IEEE standards
        result = numerator / denominator
        assert result[0] == 1.0
        assert np.isinf(result[1])
        assert result[2] == 1.5
    
    def test_log_of_negative_numbers(self, backend):
        """Test logarithm of negative numbers."""
        negative_arr = backend.array([-1.0, 0.0, 1.0])
        
        result = backend.log(negative_arr)
        assert np.isnan(result[0])  # log(-1) = NaN
        assert np.isinf(result[1]) and result[1] < 0  # log(0) = -inf
        assert result[2] == 0.0  # log(1) = 0
    
    def test_sqrt_of_negative_numbers(self, backend):
        """Test square root of negative numbers."""
        negative_arr = backend.array([-1.0, 0.0, 4.0])
        
        result = backend.sqrt(negative_arr)
        assert np.isnan(result[0])  # sqrt(-1) = NaN
        assert result[1] == 0.0
        assert result[2] == 2.0
    
    def test_power_edge_cases(self, backend):
        """Test power function edge cases."""
        base = backend.array([0.0, 1.0, -1.0, 2.0])
        
        # 0^0 case
        result_zero_zero = backend.power(base[0], 0.0)
        assert result_zero_zero == 1.0  # JAX follows the convention 0^0 = 1
        
        # Negative base with fractional exponent
        result_neg_frac = backend.power(base[2], 0.5)
        assert np.isnan(result_neg_frac)
    
    def test_exp_overflow(self, backend):
        """Test exponential function overflow behavior."""
        large_values = backend.array([700.0, 800.0, 1000.0])
        
        result = backend.exp(large_values)
        # Should handle overflow gracefully (return inf)
        assert np.all(np.isinf(result))
    
    def test_tanh_saturation(self, backend):
        """Test tanh saturation at extreme values."""
        extreme_values = backend.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        
        result = backend.tanh(extreme_values)
        
        # tanh should saturate at ±1
        np.testing.assert_almost_equal(result[0], -1.0, decimal=5)
        np.testing.assert_almost_equal(result[4], 1.0, decimal=5)
        assert result[2] == 0.0
    
    def test_erf_edge_cases(self, backend):
        """Test error function edge cases."""
        values = backend.array([-10.0, 0.0, 10.0])
        
        result = backend.erf(values)
        
        # erf should saturate at ±1
        np.testing.assert_almost_equal(result[0], -1.0, decimal=5)
        assert result[1] == 0.0
        np.testing.assert_almost_equal(result[2], 1.0, decimal=5)
    
    def test_gauss_pdf_normalization(self, backend):
        """Test Gaussian PDF normalization."""
        x = backend.linspace(-5, 5, 1000)
        mu, sigma = 0.0, 1.0
        
        pdf_values = backend.gauss_pdf(x, mu, sigma)
        
        # Numerical integration should be close to 1
        dx = 10.0 / 999  # (5 - (-5)) / (1000 - 1)
        integral = backend.sum(pdf_values) * dx
        np.testing.assert_almost_equal(integral, 1.0, decimal=2)
    
    def test_gauss_cdf_bounds(self, backend):
        """Test Gaussian CDF bounds."""
        x = backend.array([-10.0, 0.0, 10.0])
        mu, sigma = 0.0, 1.0
        
        cdf_values = backend.gauss_cdf(x, mu, sigma)
        
        # CDF should be bounded between 0 and 1
        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)
        
        # CDF at mean should be 0.5
        np.testing.assert_almost_equal(cdf_values[1], 0.5, decimal=5)
        
        # CDF should approach bounds at extremes (with relaxed tolerance for fasterf approximation)
        assert cdf_values[0] < 0.1  # Should be close to 0
        assert cdf_values[2] > 0.9  # Should be close to 1
    
    def test_histogram_empty_data(self, backend):
        """Test histogram with empty data."""
        empty_data = backend.array([])
        bins = backend.array([0, 1, 2, 3])
        weights = backend.array([])
        
        result = backend.histogram(empty_data, bins, weights)
        
        # Should return zeros for all bins
        expected = backend.zeros(len(bins) - 1)
        np.testing.assert_array_equal(result, expected)
    
    def test_bincount_edge_cases(self, backend):
        """Test bincount with edge cases."""
        # Empty array
        empty_indices = backend.array([], dtype=jnp.int32)
        empty_weights = backend.array([])
        
        result = backend.bincount(empty_indices, empty_weights, 5)
        expected = backend.zeros(5)
        np.testing.assert_array_equal(result, expected)
        
        # Single element
        single_index = backend.array([2])
        single_weight = backend.array([3.5])
        
        result = backend.bincount(single_index, single_weight, 5)
        expected = backend.array([0.0, 0.0, 3.5, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_where_mixed_types(self, backend):
        """Test where function with mixed data types."""
        condition = backend.array([True, False, True, False])
        x_values = backend.array([1.0, 2.0, 3.0, 4.0])
        y_values = backend.array([10.0, 20.0, 30.0, 40.0])
        
        result = backend.where(condition, x_values, y_values)
        expected = backend.array([1.0, 20.0, 3.0, 40.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_clip_edge_values(self, backend):
        """Test clipping at boundary values."""
        values = backend.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        low, high = -1.0, 1.0
        
        result = backend.clip(values, low, high)
        expected = backend.array([-1.0, -1.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_searchsorted_edge_cases(self, backend):
        """Test searchsorted with edge cases."""
        # Sorted array
        sorted_arr = backend.array([1.0, 3.0, 5.0, 7.0, 9.0])
        
        # Values outside range
        values = backend.array([0.0, 10.0, 5.0])
        
        # Left side
        result_left = backend.searchsorted(sorted_arr, values, side='left')
        expected_left = jnp.array([0, 5, 2])  # 0.0->index 0, 10.0->index 5, 5.0->index 2
        np.testing.assert_array_equal(result_left, expected_left)
        
        # Right side
        result_right = backend.searchsorted(sorted_arr, values, side='right')
        expected_right = jnp.array([0, 5, 3])  # 5.0->index 3 for right side
        np.testing.assert_array_equal(result_right, expected_right)
    
    def test_digitize_boundary_behavior(self, backend):
        """Test digitize behavior at bin boundaries."""
        x = backend.array([0.5, 1.0, 1.5, 2.0, 2.5])
        bins = backend.array([1.0, 2.0, 3.0])
        
        result = backend.digitize(x, bins)
        
        # Should follow JAX/NumPy convention for boundary handling
        expected = jnp.array([0, 1, 1, 2, 2])
        np.testing.assert_array_equal(result, expected)
    
    def test_ravel_multi_index_clipping(self, backend):
        """Test ravel_multi_index with out-of-bounds indices."""
        # Test with indices that would be out of bounds
        multi_index = (backend.array([0, 2, 5]), backend.array([1, 3, 1]))
        dims = (3, 4)  # 3x4 array
        
        # Should clip out-of-bounds indices
        result = backend.ravel_multi_index(multi_index, dims)
        
        # All results should be valid flat indices
        assert np.all(result >= 0)
        assert np.all(result < 3 * 4)
    
    def test_sigmoid_saturation(self, backend):
        """Test sigmoid function saturation."""
        extreme_values = backend.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        
        result = backend.sigmoid(extreme_values)
        
        # Should saturate near 0 and 1
        assert result[0] < 1e-10  # Very close to 0
        assert result[4] > 1 - 1e-6  # Very close to 1 (relaxed tolerance)
        assert abs(result[2] - 0.5) < 1e-10  # sigmoid(0) = 0.5
    
    def test_weighted_quantile_edge_cases(self, backend):
        """Test weighted quantile with edge cases."""
        # Single value
        single_x = backend.array([5.0])
        single_w = backend.array([1.0])
        
        result = backend.weighted_quantile(single_x, single_w, 0.5)
        assert result == 5.0
        
        # Zero weights (should handle gracefully)
        x = backend.array([1.0, 2.0, 3.0])
        zero_weights = backend.array([0.0, 0.0, 0.0])
        
        # This might raise an error or return a specific value depending on implementation
        try:
            result = backend.weighted_quantile(x, zero_weights, 0.5)
            # If it doesn't raise an error, result should be reasonable
            assert np.isfinite(result)
        except (ValueError, ZeroDivisionError):
            # This is also acceptable behavior
            pass
    
    def test_allclose_with_nans(self, backend):
        """Test allclose with NaN values."""
        a = backend.array([1.0, np.nan, 3.0])
        b = backend.array([1.0, np.nan, 3.0])
        
        # NaN != NaN, so this should be False
        result = backend.allclose(a, b)
        assert not result
        
        # Test without NaNs
        a_clean = backend.array([1.0, 2.0, 3.0])
        b_clean = backend.array([1.0, 2.0, 3.0])
        
        result_clean = backend.allclose(a_clean, b_clean)
        assert result_clean
    
    def test_any_with_all_false(self, backend):
        """Test any function with all False values."""
        all_false = backend.array([False, False, False])
        
        result = backend.any(all_false)
        assert not result
        
        # Test with mixed values
        mixed = backend.array([False, True, False])
        result_mixed = backend.any(mixed)
        assert result_mixed


class TestJAXBackendCompilation:
    """Test JAX compilation features."""
    
    @pytest.fixture
    def backend(self):
        """Provide a JAX backend instance."""
        return JAXBackend()
    
    def test_compile_simple_function(self, backend):
        """Test compiling a simple function."""
        def simple_func(x):
            return backend.multiply(x, 2.0)
        
        compiled_func = backend.compile(simple_func)
        
        # Test that compiled function works
        x = backend.array([1.0, 2.0, 3.0])
        result = compiled_func(x)
        expected = backend.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_grad_simple_function(self, backend):
        """Test gradient computation."""
        def quadratic(x):
            return backend.sum(x ** 2)
        
        grad_func = backend.grad(quadratic)
        
        # Gradient of sum(x^2) is 2*x
        x = backend.array([1.0, 2.0, 3.0])
        gradient = grad_func(x)
        expected = 2.0 * x
        np.testing.assert_array_almost_equal(gradient, expected)
    
    def test_func_and_grad(self, backend):
        """Test simultaneous function and gradient computation."""
        def quadratic(x):
            return backend.sum(x ** 2)
        
        func_and_grad = backend.func_and_grad(quadratic)
        
        x = backend.array([1.0, 2.0])
        value, gradient = func_and_grad(x)
        
        expected_value = 1.0 + 4.0  # 1^2 + 2^2
        expected_grad = backend.array([2.0, 4.0])  # 2*x
        
        assert value == expected_value
        np.testing.assert_array_almost_equal(gradient, expected_grad)


class TestJAXBackendMemoryEfficiency:
    """Test memory efficiency of backend operations."""
    
    @pytest.fixture
    def backend(self):
        """Provide a JAX backend instance."""
        return JAXBackend()
    
    def test_large_array_operations(self, backend):
        """Test operations on large arrays."""
        # Create moderately large arrays for testing
        n = 10000
        large_array = backend.array(np.random.rand(n))
        
        # Test that operations complete without memory issues
        result = backend.multiply(large_array, 2.0)
        assert len(result) == n
        
        # Test reduction operations
        sum_result = backend.sum(large_array)
        assert np.isfinite(sum_result)
    
    def test_chained_operations_memory(self, backend):
        """Test memory efficiency of chained operations."""
        x = backend.array(np.random.rand(1000))
        
        # Chain multiple operations
        result = backend.exp(backend.log(backend.sqrt(backend.power(x, 2))))
        
        # Should be approximately equal to original (within numerical precision)
        np.testing.assert_array_almost_equal(result, x, decimal=6)
