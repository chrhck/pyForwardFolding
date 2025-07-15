"""Tests for the backend module."""

import jax.numpy as jnp
import numpy as np
import pytest

from pyForwardFolding.backend import Array, Backend, JAXBackend


class TestBackendProtocol:
    """Test the Backend Protocol interface."""
    
    def test_jax_backend_implements_protocol(self):
        """Test that JAXBackend implements the Backend protocol."""
        backend = JAXBackend()
        
        # Test that JAXBackend instance is recognized as implementing Backend protocol
        assert isinstance(backend, Backend)
        
        # Test that jnp.ndarray is recognized as implementing Array protocol
        arr = jnp.array([1, 2, 3])
        assert isinstance(arr, Array)
    
    def test_protocol_method_signatures(self):
        """Test that all protocol methods are implemented in JAXBackend."""
        backend = JAXBackend()
        
        # Test that all methods exist and are callable
        required_methods = [
            'array', 'zeros', 'multiply', 'power', 'exp', 'sqrt', 'erf', 'fasterf',
            'gauss_pdf', 'gauss_cdf', 'uniform_pdf', 'set_index', 'fill', 
            'histogram', 'bincount', 'reshape', 'set_index_add', 'searchsorted',
            'ravel_multi_index', 'clip', 'linspace', 'where_sum', 'log', 'diff',
            'allclose', 'sum', 'tanh', 'digitize', 'any', 'where', 'sigmoid',
            'select', 'gammaln', 'func_and_grad', 'compile', 'grad', 'logspace',
            'arccos', 'arg_weighted_quantile', 'weighted_quantile', 'weighted_median'
        ]
        
        for method_name in required_methods:
            assert hasattr(backend, method_name), f"Missing method: {method_name}"
            assert callable(getattr(backend, method_name)), f"Method not callable: {method_name}"


class TestJAXBackend:
    """Test the JAX backend implementation."""
    
    @pytest.fixture
    def backend(self):
        """Provide a JAX backend instance."""
        return JAXBackend()
    
    def test_array_creation(self, backend):
        """Test array creation operations."""
        # Test array creation
        arr = backend.array([1, 2, 3])
        assert isinstance(arr, jnp.ndarray)
        np.testing.assert_array_equal(arr, jnp.array([1, 2, 3]))
        
        # Test zeros creation
        zeros = backend.zeros((3, 2))
        assert zeros.shape == (3, 2)
        np.testing.assert_array_equal(zeros, jnp.zeros((3, 2)))
        
        # Test zeros with dtype
        zeros_int = backend.zeros((2, 2), dtype=int)
        assert zeros_int.dtype in (int, jnp.int32, jnp.int64)  # Accept various int dtypes
    
    def test_arithmetic_operations(self, backend):
        """Test basic arithmetic operations."""
        a = backend.array([1., 2., 3.])
        b = backend.array([2., 3., 4.])
        
        # Test multiplication
        result = backend.multiply(a, b)
        expected = jnp.array([2., 6., 12.])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test power
        result = backend.power(a, 2.0)
        expected = jnp.array([1., 4., 9.])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test sum
        result = backend.sum(a)
        assert result == 6.0
        
        # Test sum with axis
        matrix = backend.array([[1., 2.], [3., 4.]])
        result = backend.sum(matrix, axis=0)
        expected = jnp.array([4., 6.])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_mathematical_functions(self, backend):
        """Test mathematical functions."""
        x = backend.array([0., 1., 2.])
        
        # Test exp
        result = backend.exp(x)
        expected = jnp.exp(x)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test sqrt
        result = backend.sqrt(backend.array([1., 4., 9.]))
        expected = jnp.array([1., 2., 3.])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test log
        result = backend.log(backend.array([1., np.e, np.e**2]))
        expected = jnp.array([0., 1., 2.])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test tanh
        result = backend.tanh(x)
        expected = jnp.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_statistical_functions(self, backend):
        """Test statistical and probability functions."""
        x = backend.array([-1., 0., 1.])
        
        # Test erf
        result = backend.erf(x)
        expected = jnp.array([-0.8427007929497149, 0., 0.8427007929497149])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        
        # Test Gaussian PDF
        result = backend.gauss_pdf(x, 0.0, 1.0)
        expected = jnp.exp(-0.5 * x**2) / jnp.sqrt(2 * jnp.pi)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test Gaussian CDF
        result = backend.gauss_cdf(x, 0.0, 1.0)
        # Compare with expected values computed using the same fasterf approximation
        expected = 0.5 * (1 + backend.fasterf(x / backend.sqrt(2.0)))
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test uniform PDF
        result = backend.uniform_pdf(backend.array([0.5, 1.5, 2.5]), 0.0, 2.0)
        expected = jnp.array([0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_array_manipulation(self, backend):
        """Test array manipulation functions."""
        # Test reshape
        arr = backend.array([1, 2, 3, 4, 5, 6])
        reshaped = backend.reshape(arr, (2, 3))
        assert reshaped.shape == (2, 3)
        
        # Test clip
        arr = backend.array([-1., 0.5, 2.])
        clipped = backend.clip(arr, 0.0, 1.0)
        expected = jnp.array([0., 0.5, 1.])
        np.testing.assert_array_almost_equal(clipped, expected)
        
        # Test diff
        arr = backend.array([1., 3., 6., 10.])
        diff_result = backend.diff(arr)
        expected = jnp.array([2., 3., 4.])
        np.testing.assert_array_almost_equal(diff_result, expected)
    
    def test_search_and_binning(self, backend):
        """Test search and binning operations."""
        # Test searchsorted
        arr = backend.array([1., 2., 3., 4.])
        values = backend.array([1.5, 2.5, 3.5])
        result = backend.searchsorted(arr, values)
        expected = jnp.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)
        
        # Test digitize
        bins = backend.array([0., 1., 2., 3.])
        values = backend.array([0.5, 1.5, 2.5])
        result = backend.digitize(values, bins)
        expected = jnp.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)
    
    def test_histogram(self, backend):
        """Test histogram functionality."""
        x = backend.array([0.5, 1.5, 2.5, 1.2, 0.8])
        bins = backend.array([0., 1., 2., 3.])
        weights = backend.array([1., 1., 1., 1., 1.])
        
        hist = backend.histogram(x, bins, weights)
        # Should have 2 entries in first bin, 2 in second, 1 in third
        expected = jnp.array([2., 2., 1.])
        np.testing.assert_array_equal(hist, expected)
    
    def test_conditional_operations(self, backend):
        """Test conditional operations."""
        condition = backend.array([True, False, True])
        x = backend.array([1., 2., 3.])
        y = backend.array([10., 20., 30.])
        
        # Test select/where
        result = backend.select(condition, x, y)
        expected = jnp.array([1., 20., 3.])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test where
        result = backend.where(condition, x, y)
        expected = jnp.array([1., 20., 3.])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_comparison_operations(self, backend):
        """Test comparison operations."""
        a = backend.array([1., 2., 3.])
        b = backend.array([1., 2.1, 2.9])
        
        # Test allclose
        assert backend.allclose(a, b, atol=0.2)
        assert not backend.allclose(a, b, atol=0.05)
    
    def test_jax_specific_functions(self, backend):
        """Test JAX-specific functionality."""
        def simple_func(x):
            return backend.sum(x**2)
        
        # Test func_and_grad
        func_and_grad = backend.func_and_grad(simple_func)
        x = backend.array([1., 2., 3.])
        value, grad = func_and_grad(x)
        
        expected_value = 14.0  # 1^2 + 2^2 + 3^2
        expected_grad = jnp.array([2., 4., 6.])  # 2*x
        
        assert abs(value - expected_value) < 1e-6
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
        # Test compile
        compiled_func = backend.compile(simple_func)
        result = compiled_func(x)
        assert abs(result - expected_value) < 1e-6
        
        # Test grad
        grad_func = backend.grad(simple_func)
        grad_result = grad_func(x)
        np.testing.assert_array_almost_equal(grad_result, expected_grad)
    
    def test_type_compatibility(self, backend):
        """Test type compatibility between different array types and protocols."""
        backend = JAXBackend()
        
        # Test that we can use the backend with type annotations
        def process_data(backend_impl: Backend[jnp.ndarray], data: jnp.ndarray) -> jnp.ndarray:
            return backend_impl.exp(backend_impl.multiply(data, data))
        
        # This should work without type errors
        test_data = backend.array([1., 2., 3.])
        result = process_data(backend, test_data)
        expected = jnp.exp(test_data * test_data)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_protocol_runtime_checking(self, backend):
        """Test runtime protocol checking capabilities."""
        backend = JAXBackend()
        
        # Test isinstance checks work with protocols
        assert isinstance(backend, Backend)
        
        # Test that jnp arrays satisfy the Array protocol
        arr = jnp.array([1, 2, 3])
        assert isinstance(arr, Array)
        
        # Test that regular Python lists don't satisfy the Array protocol
        python_list = [1, 2, 3]
        assert not isinstance(python_list, Array)


class TestBackendFlexibility:
    """Test the flexibility of the Protocol-based backend system."""
    
    def test_backend_agnostic_functions(self):
        """Test that functions can work with any backend that implements the protocol."""
        
        def compute_gaussian_sum(backend_impl: Backend, x, mu: float = 0.0, sigma: float = 1.0):
            """Example function that works with any backend."""
            pdf_values = backend_impl.gauss_pdf(x, mu, sigma)
            return backend_impl.sum(pdf_values)
        
        # Test with JAX backend
        jax_backend = JAXBackend()
        x = jax_backend.array([-1., 0., 1.])
        result = compute_gaussian_sum(jax_backend, x)
        
        # Verify the result makes sense
        assert isinstance(result, jnp.ndarray)
        assert result > 0  # Sum of PDF values should be positive
    
    def test_backend_method_chaining(self):
        """Test complex operations using method chaining."""
        backend = JAXBackend()
        
        # Complex computation using multiple backend methods
        x = backend.array([1., 2., 3., 4., 5.])
        
        # Chain multiple operations: (exp(x) - 1) / x, then clip, then sum
        result = backend.sum(
            backend.clip(
                (backend.exp(x) - 1) / x,
                0.5, 
                10.0
            )
        )
        
        # Verify the result is reasonable
        assert isinstance(result, jnp.ndarray)
        assert result > 0
