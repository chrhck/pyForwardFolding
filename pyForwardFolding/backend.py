from math import pi
from typing import Any, Protocol, Tuple, TypeVar, cast

import jax.nn
import jax.numpy as jnp
import jax.scipy.special
from jax import Array as JAXArray
from jax.typing import ArrayLike
from typing_extensions import runtime_checkable


@runtime_checkable
class Array(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]: ...
    
    @property
    def dtype(self) -> Any: ...

    def __getitem__(self, key) -> "Array": ...
    def __setitem__(self, key, value) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __reversed__(self) -> "Array": ...
    def __round__(self, ndigits=None) -> "Array": ...

    # Comparisons

    # these return bool for object, so ignore override errors.
    def __lt__(self, other) -> "Array": ...
    def __le__(self, other) -> "Array": ...
    def __eq__(self, other) -> "Array": ...  # type: ignore[override]
    def __ne__(self, other) -> "Array": ...  # type: ignore[override]
    def __gt__(self, other) -> "Array": ...
    def __ge__(self, other) -> "Array": ...

    # Unary arithmetic

    def __neg__(self) -> "Array": ...
    def __pos__(self) -> "Array": ...
    def __abs__(self) -> "Array": ...
    def __invert__(self) -> "Array": ...

    # Binary arithmetic

    def __add__(self, other) -> "Array": ...
    def __sub__(self, other) -> "Array": ...
    def __mul__(self, other) -> "Array": ...
    def __matmul__(self, other) -> "Array": ...
    def __truediv__(self, other) -> "Array": ...
    def __floordiv__(self, other) -> "Array": ...
    def __mod__(self, other) -> "Array": ...
    def __divmod__(self, other) -> tuple["Array", "Array"]: ...
    def __pow__(self, other) -> "Array": ...
    def __lshift__(self, other) -> "Array": ...
    def __rshift__(self, other) -> "Array": ...
    def __and__(self, other) -> "Array": ...
    def __xor__(self, other) -> "Array": ...
    def __or__(self, other) -> "Array": ...

    def __radd__(self, other) -> "Array": ...
    def __rsub__(self, other) -> "Array": ...
    def __rmul__(self, other) -> "Array": ...
    def __rmatmul__(self, other) -> "Array": ...
    def __rtruediv__(self, other) -> "Array": ...
    def __rfloordiv__(self, other) -> "Array": ...
    def __rmod__(self, other) -> "Array": ...
    def __rdivmod__(self, other) -> "Array": ...
    def __rpow__(self, other) -> "Array": ...
    def __rlshift__(self, other) -> "Array": ...
    def __rrshift__(self, other) -> "Array": ...
    def __rand__(self, other) -> "Array": ...
    def __rxor__(self, other) -> "Array": ...
    def __ror__(self, other) -> "Array": ...

    def __bool__(self) -> bool: ...
    def __complex__(self) -> complex: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __index__(self) -> int: ...

    def __buffer__(self, flags: int) -> memoryview: ...
    
    
    def astype(self, dtype: Any) -> "Array": ...

ArrayType = TypeVar('ArrayType', bound=Array)

@runtime_checkable
class Backend(Protocol[ArrayType]):
    """
    Protocol for backend interface for numerical operations.
    """

    def array(self, data: Any, dtype: Any = None) -> ArrayType:
        """Create an array from data."""
        ...

    def asarray(self, data: Any, dtype: Any = None) -> ArrayType:
        """Create an array from data without copying if possible."""
        ...
    
    
    def zeros(self, shape: Any, dtype: Any = None) -> ArrayType:
        """Create an array of zeros."""
        ...

    def power(self, a: Any, b: Any) -> ArrayType:
        """Element-wise power operation."""
        ...

    def exp(self, a: ArrayType) -> ArrayType:
        """Element-wise exponential."""
        ...

    def sqrt(self, a: ArrayType) -> ArrayType:
        """Element-wise square root."""
        ...

    def erf(self, a: ArrayType) -> ArrayType:
        """Element-wise error function."""
        ...
    
    def fasterf(self, x: ArrayType) -> ArrayType:
        """Fast approximation of error function."""
        ...

    def gauss_pdf(self, x: Any, mu: Any, sigma: Any) -> ArrayType:
        """Gaussian probability density function."""
        ...

    def gauss_cdf(self, x: Any, mu: Any, sigma: Any) -> ArrayType:
        """Gaussian cumulative distribution function."""
        ...

    def uniform_pdf(self, x: Any, lo: Any, hi: Any) -> ArrayType:
        """Uniform probability density function."""
        ...

    def set_index(self, x: ArrayType, index: Any, values: Any) -> ArrayType:
        """Set values at specified indices."""
        ...
    
    def fill(self, x: ArrayType, value: Any) -> ArrayType:
        """Fill array with a value."""
        ...

    def histogram(self, x: ArrayType, bins: Any, weights: ArrayType) -> ArrayType:
        """Compute histogram."""
        ...
    
    def bincount(self, x: ArrayType, weights: ArrayType, length: int) -> ArrayType:
        """Count occurrences of each value in array."""
        ...
    
    def reshape(self, x: ArrayType, shape: Any) -> ArrayType:
        """Reshape array."""
        ...
    
    def set_index_add(self, x: ArrayType, index: Any, values: Any) -> ArrayType:
        """Add values at specified indices."""
        ...
    
    def searchsorted(self, a: ArrayType, v: ArrayType, side: str = 'left') -> ArrayType:
        """Find indices where elements should be inserted to maintain order."""
        ...

    def ravel_multi_index(self, multi_index: Any, dims: Any) -> ArrayType:
        """
        Converts a multi-dimensional index into a flat index.
        """
        ...
    
    def clip(self, x: ArrayType, low: Any, high: Any) -> ArrayType:
        """
        Clamps the values in x to be within the range [low, high].
        """
        ...
    
    def linspace(self, start: float, stop: float, num: int) -> ArrayType:
        """
        Returns evenly spaced numbers over a specified interval.
        """
        ...
    
    def where_sum(self, condition: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType:
        """
        Returns the sum of an array with elements from x where condition is True, and from y otherwise.
        """
        ...
    
    def log(self, x: Any) -> ArrayType:
        """
        Computes the natural logarithm of x.
        """
        ...
    
    def diff(self, x: Any) -> ArrayType:
        """
        Computes the discrete difference along the specified axis.
        """
        ...

    def allclose(self, a: ArrayType, b: ArrayType, atol: float = 1e-8) -> ArrayType:
        """
        Checks if two arrays are element-wise equal within a tolerance.
        """
        ...

    def sum(self, x: ArrayType, axis: Any = None) -> ArrayType:
        """
        Computes the sum of array elements over a given axis.
        """
        ...
    
    def tanh(self, x: ArrayType) -> ArrayType:
        """
        Computes the hyperbolic tangent of x.
        """
        ...

    def digitize(self, x: ArrayType, bins: ArrayType) -> ArrayType:
        """
        Returns the indices of the bins to which each value in x belongs.
        """
        ...
    
    def any(self, x: ArrayType) -> ArrayType:
        """
        Returns True if any element in x is True
        """
        ...
    
    def where(self, cond: Any, x: Any, y: Any) -> ArrayType:
        """
        Depending on cond return value from x or y
        """
        ...

    def sigmoid(self, x: ArrayType) -> ArrayType:
        """
        Sigmoid function
        """
        ...

    def select(self, condition: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType:
        """
        Select elements from x or y depending on condition.
        """
        ...
    
    def gammaln(self, x: ArrayType) -> ArrayType:
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
    
    def logspace(self, start: float, stop: float, num: int) -> ArrayType:
        """
        Return numbers spaced evenly on a log scale.
        """
        ...
    
    def arccos(self, x: ArrayType) -> ArrayType:
        """
        Trigonometric inverse cosine, element-wise.
        """
        ...
    
    def arg_weighted_quantile(self, x: ArrayType, weights: ArrayType, quantile: float) -> ArrayType:
        """
        Return the index of the weighted quantile.
        """
        ...
    
    def weighted_quantile(self, x: ArrayType, weights: ArrayType, quantile: float) -> ArrayType:
        """
        Return the weighted quantile.
        """
        ...
    
    def weighted_median(self, x: ArrayType, weights: ArrayType) -> ArrayType:
        """
        Return the weighted median.
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
        return a ** b

    def exp(self, a: ArrayLike) -> JAXArray:
        return jnp.exp(a)

    def sqrt(self, a: ArrayLike) -> JAXArray:
        return jnp.sqrt(a)

    def erf(self, a: ArrayLike) -> JAXArray:
        return jax.scipy.special.erf(a)
    
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

        erf_pos = 1 - (a_1*t + a_2*t**2 + a_3*t**3) * jnp.exp(-x_abs**2)
        
        # Apply sign correction for negative values
        return jnp.where(x >= 0, erf_pos, -erf_pos)


    def gauss_pdf(self, x: ArrayLike, mu: Any, sigma: Any) -> JAXArray:
        return 1. / (sigma * self.sqrt(2 * pi)) * self.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def gauss_cdf(self, x: ArrayLike, mu: Any, sigma: Any) -> JAXArray:
        return 0.5 * (1. + self.fasterf((x - mu) / (sigma * self.sqrt(2.))))

    def uniform_pdf(self, x: ArrayLike, lo: Any, hi: Any) -> JAXArray:
        # Uniform PDF is 1/(hi-lo) if lo <= x <= hi, 0 otherwise
        pdf_value = 1. / (hi - lo)
        return cast(JAXArray, jnp.where((x >= lo) & (x <= hi), pdf_value, 0.0))
    
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
        hist, _ = jnp.histogram(x, bins=bins, weights=weights)
        return hist
    
    def bincount(self, x: ArrayLike, weights: ArrayLike, length: int) -> JAXArray:
        return jnp.bincount(x, weights=weights, length=length)
    
    def reshape(self, x: ArrayLike, shape: Any) -> JAXArray:
        x = jnp.asarray(x)
        return x.reshape(shape)
    
    def searchsorted(self, a: ArrayLike, v: ArrayLike, side: str = 'left') -> JAXArray:
        return jnp.searchsorted(a, v, side=side)
    
    def ravel_multi_index(self, multi_index: Any, dims: Any) -> JAXArray:
        return jnp.ravel_multi_index(multi_index, dims, mode="clip")
    
    def clip(self, x: ArrayLike, low: Any, high: Any) -> JAXArray:
        return jnp.clip(x, low, high)
    
    def linspace(self, start: float, stop: float, num: int) -> JAXArray:
        return jnp.linspace(start, stop, num)
    
    def where_sum(self, condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> JAXArray:
        return jnp.where(condition, x, y).sum()
    
    def log(self, x: ArrayLike) -> JAXArray:
        return jnp.log(x)
    
    def diff(self, x: ArrayLike, axis: int = 0) -> JAXArray:
        return jnp.diff(x, axis=axis)
    
    def allclose(self, a: ArrayLike, b: ArrayLike, atol: float = 1e-8) -> JAXArray:
        return jnp.allclose(a, b, atol=atol)
    
    def sum(self, x: ArrayLike, axis: Any = None) -> JAXArray:
        return jnp.sum(x, axis=axis)
    
    def tanh(self, x: ArrayLike) -> JAXArray:
        return jnp.tanh(x)

    def select(self, condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> JAXArray:
        return jnp.where(condition, x, y)
    
    def gammaln(self, x: ArrayLike) -> JAXArray:
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
        return jnp.arccos(x)
    
    def digitize(self, x: ArrayLike, bins: ArrayLike) -> JAXArray:
        return jnp.digitize(x, bins, right=False)
    
    def arg_weighted_quantile(self, x: ArrayLike, weights: ArrayLike, quantile: float) -> JAXArray:
        if not (0 <= quantile <= 1):
            raise ValueError("quantile must have a value between 0 and 1")

        x = jnp.asarray(x)
        sorted_indices = jnp.argsort(x)
        weights = jnp.asarray(weights)
        sorted_weightes = weights[sorted_indices]
        cumul_weights = jnp.cumsum(sorted_weightes) / jnp.sum(weights)
        quantile_idx = jnp.searchsorted(cumul_weights, quantile)

        quantile_idx = jnp.clip(quantile_idx, 0, len(x) - 1)

        return sorted_indices[quantile_idx]
    
    
    def weighted_quantile(self, x: ArrayLike, weights: ArrayLike, quantile: float) -> JAXArray:
        x = jnp.asarray(x)
        return x[self.arg_weighted_quantile(x, weights, quantile)]
    
    
    def weighted_median(self, x: ArrayLike, weights: ArrayLike) -> JAXArray:
        return self.weighted_quantile(x, weights, 0.5)
    
    def any(self, x: ArrayLike) -> JAXArray:
        return jnp.any(x)

    def where(self, cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> JAXArray:
        return jnp.where(cond, x, y)

    def sigmoid(self, x: ArrayLike) -> JAXArray:
        return jax.nn.sigmoid(x)

# Type aliases for convenience
JAXBackendType = Backend[JAXArray]

# Default backend instance
backend: JAXBackendType = JAXBackend()

