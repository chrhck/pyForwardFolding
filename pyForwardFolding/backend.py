from math import pi
from typing import Any, Protocol, Union

import jax.nn
import jax.numpy as jnp
import jax.scipy.special
from jax import Array as JAXArray
from jax.typing import ArrayLike as JArrayLike
from typing_extensions import runtime_checkable

# Base types that backends should work with
Array = JAXArray
ArrayLike = Union[JAXArray, JArrayLike, Any]  # Allow any array-like input


@runtime_checkable
class Backend(Protocol):
    """
    Protocol for backend interface for numerical operations.
    """

    def array(self, data: Any, dtype: Any = None) -> Array:
        """Create an array from data."""
        ...

    def asarray(self, data: Any, dtype: Any = None) -> Array:
        """Create an array from data without copying if possible."""
        ...

    def zeros(self, shape: Any, dtype: Any = None) -> Array:
        """Create an array of zeros."""
        ...

    def power(self, a: Any, b: Any) -> Array:
        """Element-wise power operation."""
        ...

    def exp(self, a: Any) -> Array:
        """Element-wise exponential."""
        ...

    def sqrt(self, a: Any) -> Array:
        """Element-wise square root."""
        ...

    def erf(self, a: Any) -> Array:
        """Element-wise error function."""
        ...

    def fasterf(self, x: Any) -> Array:
        """Fast approximation of error function."""
        ...

    def gauss_pdf(self, x: Any, mu: Any, sigma: Any) -> Array:
        """Gaussian probability density function."""
        ...

    def gauss_cdf(self, x: Any, mu: Any, sigma: Any) -> Array:
        """Gaussian cumulative distribution function."""
        ...

    def uniform_pdf(self, x: Any, lo: Any, hi: Any) -> Array:
        """Uniform probability density function."""
        ...

    def set_index(self, x: ArrayLike, index: Any, values: Any) -> Array:
        """Set values at specified indices."""
        ...

    def fill(self, x: ArrayLike, value: Any) -> Array:
        """Fill array with a value."""
        ...

    def histogram(self, x: Any, bins: Any, weights: Any) -> Array:
        """Compute histogram."""
        ...

    def bincount(self, x: Array, weights: Array, length: int) -> Array:
        """Count occurrences of each value in array."""
        ...

    def reshape(self, x: Array, shape: Any) -> Array:
        """Reshape array."""
        ...

    def set_index_add(self, x: Array, index: Any, values: Any) -> Array:
        """Add values at specified indices."""
        ...

    def searchsorted(self, a: Array, v: Array, side: str = "left") -> Array:
        """Find indices where elements should be inserted to maintain order."""
        ...

    def ravel_multi_index(self, multi_index: Any, dims: Any) -> Array:
        """
        Converts a multi-dimensional index into a flat index.
        """
        ...

    def clip(self, x: Array, low: Any, high: Any) -> Array:
        """
        Clamps the values in x to be within the range [low, high].
        """
        ...

    def linspace(self, start: float, stop: float, num: int) -> Array:
        """
        Returns evenly spaced numbers over a specified interval.
        """
        ...

    def where_sum(self, condition: Array, x: Array, y: ArrayLike) -> ArrayLike:
        """
        Returns the sum of an array with elements from x where condition is True, and from y otherwise.
        """
        ...

    def log(self, x: Any) -> Array:
        """
        Computes the natural logarithm of x.
        """
        ...

    def diff(self, x: Any) -> Array:
        """
        Computes the discrete difference along the specified axis.
        """
        ...

    def allclose(self, a: Any, b: Any, atol: float = 1e-8) -> Array:
        """
        Checks if two arrays are element-wise equal within a tolerance.
        """
        ...

    def sum(self, x: Array, axis: Any = None) -> Array:
        """
        Computes the sum of array elements over a given axis.
        """
        ...

    def tanh(self, x: Any) -> Any:
        """
        Computes the hyperbolic tangent of x.
        """
        ...

    def digitize(self, x: Array, bins: Array) -> Array:
        """
        Returns the indices of the bins to which each value in x belongs.
        """
        ...

    def any(self, x: Array) -> Array:
        """
        Returns True if any element in x is True
        """
        ...

    def where(self, cond: Any, x: Any, y: Any) -> Array:
        """
        Depending on cond return value from x or y
        """
        ...

    def sigmoid(self, x: Array) -> Array:
        """
        Sigmoid function
        """
        ...

    def select(self, condition: Array, x: ArrayLike, y: ArrayLike) -> Array:
        """
        Select elements from x or y depending on condition.
        """
        ...

    def gammaln(self, x: Array) -> Array:
        """
        Natural logarithm of the gamma function.
        """
        ...

    def func_and_grad(self, func: Any) -> Any:
        """
        Return a function that computes both value and gradient.
        """
        ...

    def compile(self, func: Any) -> Any:
        """
        Compile a function for faster execution.
        """
        ...

    def grad(self, func: Any) -> Any:
        """
        Return the gradient of a function.
        """
        ...

    def logspace(self, start: float, stop: float, num: int) -> Array:
        """
        Return numbers spaced evenly on a log scale.
        """
        ...

    def arccos(self, x: Array) -> Array:
        """
        Trigonometric inverse cosine, element-wise.
        """
        ...

    def arg_weighted_quantile(self, x: Array, weights: Array, quantile: float) -> Array:
        """
        Return the index of the weighted quantile.
        """
        ...

    def weighted_quantile(self, x: Array, weights: Array, quantile: float) -> Array:
        """
        Return the weighted quantile.
        """
        ...

    def weighted_median(self, x: Array, weights: Array) -> Array:
        """
        Return the weighted median.
        """
        ...

    def repeat(self, x: Array, repeats, axis=None, total_repeat_length=None) -> Array:
        """
        Repeat elements of an array.
        """
        ...


class JAXBackend:
    """
    JAX implementation of the backend interface.

    Note: This class doesn't need to inherit from Backend protocol
    because it uses structural typing - any class that implements
    all the required methods automatically satisfies the protocol.
    """

    def array(self, data: Any, dtype: Any = None) -> JAXArray:
        return jnp.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any = None) -> JAXArray:
        return jnp.asarray(data, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> JAXArray:
        return jnp.zeros(shape, dtype=dtype)

    def power(self, a: ArrayLike, b: Any) -> JAXArray:
        return jnp.power(jnp.asarray(a), b)

    def exp(self, a: ArrayLike) -> JAXArray:
        return jnp.exp(jnp.asarray(a))

    def sqrt(self, a: ArrayLike) -> JAXArray:
        return jnp.sqrt(jnp.asarray(a))

    def erf(self, a: ArrayLike) -> JAXArray:
        return jax.scipy.special.erf(jnp.asarray(a))

    def fasterf(self, x: ArrayLike) -> JAXArray:
        # Approximation of erf function that handles negative values
        # erf(-x) = -erf(x), so we can use the absolute value
        x = jnp.asarray(x)
        x_abs = jnp.abs(x)

        # Handle extreme values to prevent overflow
        x_abs = jnp.clip(x_abs, 0, 10)  # Clip to reasonable range

        p = 0.47047
        t = 1 / (1 + p * x_abs)

        a_1 = 0.3480242
        a_2 = -0.0958798
        a_3 = 0.7478556

        erf_pos = 1 - (a_1 * t + a_2 * t**2 + a_3 * t**3) * jnp.exp(-(x_abs**2))

        # Apply sign correction for negative values
        return jnp.where(x >= 0, erf_pos, -erf_pos)

    def gauss_pdf(self, x: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        mu = jnp.asarray(mu)
        sigma = jnp.asarray(sigma)
        return (
            1.0 / (sigma * self.sqrt(2 * pi)) * self.exp(-0.5 * ((x - mu) / sigma) ** 2)
        )

    def gauss_cdf(self, x: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        mu = jnp.asarray(mu)
        sigma = jnp.asarray(sigma)
        return 0.5 * (1.0 + self.fasterf((x - mu) / (sigma * self.sqrt(2.0))))

    def uniform_pdf(self, x: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> JAXArray:
        # Uniform PDF is 1/(hi-lo) if lo <= x <= hi, 0 otherwise
        x = jnp.asarray(x)
        lo = jnp.asarray(lo)
        hi = jnp.asarray(hi)
        pdf_value = 1.0 / (hi - lo)
        result = jnp.where((x >= lo) & (x <= hi), pdf_value, 0.0)
        # Ensure we return a JAXArray, not a tuple
        if isinstance(result, tuple):
            return result[0]
        return result

    def set_index(self, x: ArrayLike, index: Any, values: Any) -> JAXArray:
        x = jnp.asarray(x)
        return x.at[index].set(values)

    def set_index_add(self, x: ArrayLike, index: Any, values: Any) -> JAXArray:
        x = jnp.asarray(x)
        return x.at[index].add(values)

    def fill(self, x: ArrayLike, value: Any) -> JAXArray:
        x = jnp.asarray(x)
        return x.at[:].set(value)

    def histogram(self, x: ArrayLike, bins: Any, weights: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        weights = jnp.asarray(weights)
        hist, _ = jnp.histogram(x, bins=bins, weights=weights)
        return hist

    def bincount(self, x: ArrayLike, weights: ArrayLike, length: int) -> JAXArray:
        x = jnp.asarray(x)
        weights = jnp.asarray(weights)
        return jnp.bincount(x, weights=weights, length=length)

    def reshape(self, x: ArrayLike, shape: Any) -> JAXArray:
        x = jnp.asarray(x)
        return x.reshape(shape)

    def searchsorted(self, a: ArrayLike, v: ArrayLike, side: str = "left") -> JAXArray:
        a = jnp.asarray(a)
        v = jnp.asarray(v)
        return jnp.searchsorted(a, v, side=side)

    def ravel_multi_index(self, multi_index: Any, dims: Any) -> JAXArray:
        return jnp.ravel_multi_index(multi_index, dims, mode="clip")

    def clip(self, x: ArrayLike, low: Any, high: Any) -> JAXArray:
        x = jnp.asarray(x)
        return jnp.clip(x, low, high)

    def linspace(self, start: float, stop: float, num: int) -> JAXArray:
        return jnp.linspace(start, stop, num)

    def where_sum(self, condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> JArrayLike:
        condition = jnp.asarray(condition)
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        wsum = jnp.where(condition, x, y).sum()
        return wsum

    def log(self, x: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        return jnp.log(x)

    def diff(self, x: ArrayLike, axis: int = 0) -> JAXArray:
        x = jnp.asarray(x)
        return jnp.diff(x, axis=axis)

    def allclose(self, a: ArrayLike, b: ArrayLike, atol: float = 1e-8) -> JAXArray:
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        return jnp.allclose(a, b, atol=atol)

    def sum(self, x: ArrayLike, axis: Any = None) -> JAXArray:
        x = jnp.asarray(x)
        return jnp.sum(x, axis=axis)

    def tanh(self, x: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        return jnp.tanh(x)

    def select(self, condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> JAXArray:
        condition = jnp.asarray(condition)
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return jnp.where(condition, x, y)

    def gammaln(self, x: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        return jax.scipy.special.gammaln(x)

    def func_and_grad(self, func: Any) -> Any:
        return jax.jit(jax.value_and_grad(func))

    def compile(self, func: Any) -> Any:
        return jax.jit(func)

    def grad(self, func: Any) -> Any:
        return jax.grad(func)

    def logspace(self, start: float, stop: float, num: int) -> JAXArray:
        return jnp.logspace(start, stop, num)

    def arccos(self, x: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        return jnp.arccos(x)

    def digitize(self, x: ArrayLike, bins: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        bins = jnp.asarray(bins)
        return jnp.digitize(x, bins, right=False)

    def arg_weighted_quantile(
        self, x: ArrayLike, weights: ArrayLike, quantile: float
    ) -> JAXArray:
        if not (0 <= quantile <= 1):
            raise ValueError("quantile must have a value between 0 and 1")

        x = jnp.asarray(x)
        weights = jnp.asarray(weights)
        sorted_indices = jnp.argsort(x)
        sorted_weightes = weights[sorted_indices]
        cumul_weights = jnp.cumsum(sorted_weightes) / jnp.sum(weights)
        quantile_idx = jnp.searchsorted(cumul_weights, quantile)

        quantile_idx = jnp.clip(quantile_idx, 0, len(x) - 1)

        return sorted_indices[quantile_idx]

    def weighted_quantile(
        self, x: ArrayLike, weights: ArrayLike, quantile: float
    ) -> JAXArray:
        x = jnp.asarray(x)
        return x[self.arg_weighted_quantile(x, weights, quantile)]

    def weighted_median(self, x: ArrayLike, weights: ArrayLike) -> JAXArray:
        return self.weighted_quantile(x, weights, 0.5)

    def any(self, x: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        return jnp.any(x)

    def where(self, cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> JAXArray:
        cond = jnp.asarray(cond)
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return jnp.where(cond, x, y)

    def sigmoid(self, x: ArrayLike) -> JAXArray:
        x = jnp.asarray(x)
        return jnp.asarray(jax.nn.sigmoid(x))


    def repeat(self, x: ArrayLike, repeats, axis=None, total_repeat_length=None) -> JAXArray:
        """
        Repeat elements of an array.
        """
        x = jnp.asarray(x)
        return jnp.repeat(x, repeats, axis=axis, total_repeat_length=total_repeat_length)

# Type aliases for convenience
JAXBackendType = Backend

# Default backend instance
backend: Backend = JAXBackend()
