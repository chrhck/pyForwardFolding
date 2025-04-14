from typing import Any
import jax.numpy as jnp
from math import sqrt, pi, exp
import jax.scipy.special


class Backend:
    """
    Abstract backend interface for numerical operations.
    """
    def array(self, data: Any) -> Any:
        raise NotImplementedError
    
    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        raise NotImplementedError

    def multiply(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def power(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def exp(self, a: Any) -> Any:
        raise NotImplementedError

    def sqrt(self, a: Any) -> Any:
        raise NotImplementedError

    def erf(self, a: Any) -> Any:
        raise NotImplementedError
    
    def fasterf(self, a) -> Any:
        raise NotImplementedError

    def gauss_pdf(self, x: Any, mu: Any, sigma: Any) -> Any:
        raise NotImplementedError

    def gauss_cdf(self, x: Any, mu: Any, sigma: Any) -> Any:
        raise NotImplementedError

    def uniform_pdf(self, x: Any, lo: Any, hi: Any) -> Any:
        raise NotImplementedError

    def set_index(self, x: Any, index: Any, values: Any) -> None:
        raise NotImplementedError
    
    def fill(self, x: Any, value: Any) -> None:
        raise NotImplementedError

    def histogram(self, x: Any, bins: Any, weights: Any) -> Any:
        raise NotImplementedError
    
    def bincount(self, x: Any, weights: Any, length:int) -> Any:
        raise NotImplementedError
    
    def reshape(self, x: Any, shape: Any) -> Any:
        raise NotImplementedError
    
    def set_index_add(self, x: Any, index: Any, values: Any) -> None:
        raise NotImplementedError
    
    def searchsorted(self, a: Any, v: Any, side: str = 'left') -> Any:
        raise NotImplementedError
    

    def ravel_multi_index(self, multi_index: Any, dims: Any) -> Any:
        """
        Converts a multi-dimensional index into a flat index.
        """
        raise NotImplementedError
    
    def clip(self, x: Any, low: Any, high: Any) -> Any:
        """
        Clamps the values in x to be within the range [low, high].
        """
        raise NotImplementedError
    
    def linspace(self, start: float, stop: float, num: int) -> Any:
        """
        Returns evenly spaced numbers over a specified interval.
        """
        raise NotImplementedError
    
    def where_sum(self, condition: Any, x: Any, y: Any) -> Any:
        """
        Returns the sum of an array with elements from x where condition is True, and from y otherwise.
        """
        raise NotImplementedError
    
    def log(self, x: Any) -> Any:
        """
        Computes the natural logarithm of x.
        """
        raise NotImplementedError
    
    def diff(self, x: Any) -> Any:
        """
        Computes the discrete difference along the specified axis.
        """
        raise NotImplementedError

    def allclose(self, a: Any, b: Any, atol: float = 1e-8) -> bool:
        """
        Checks if two arrays are element-wise equal within a tolerance.
        """
        raise NotImplementedError

    def sum(self, x: Any, axis: Any = None) -> Any:
        """
        Computes the sum of array elements over a given axis.
        """
        raise NotImplementedError
    
    def tanh(self, x: Any) -> Any:
        """
        Computes the hyperbolic tangent of x.
        """
        raise NotImplementedError

class JAXBackend(Backend):
    """
    JAX implementation of the backend interface.
    """
    def array(self, data: Any) -> jnp.ndarray:
        return jnp.array(data)
    
    def zeros(self, shape: Any, dtype: Any = None) -> jnp.ndarray:
        return jnp.zeros(shape, dtype=dtype)

    def multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a * b

    def power(self, a: jnp.ndarray, b: float) -> jnp.ndarray:
        return a ** b

    def exp(self, a: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(a)

    def sqrt(self, a: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(a)

    def erf(self, a: jnp.ndarray) -> jnp.ndarray:
        return jnp.scipy.special.erf(a)
    
    def fasterf(self, x: jnp.ndarray) -> jnp.ndarray:
        p = 0.47047
        t = 1 / (1 + p*x)

        a_1 = 0.3480242
        a_2 = -0.0958798
        a_3 = 0.7478556

        return 1 - (a_1*t +a_2*t**2 + a_3*t**3)*jnp.exp(-x**2)


    def gauss_pdf(self, x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
        return 1. / (sigma * self.sqrt(2 * pi)) * self.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def gauss_cdf(self, x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
        return 0.5 * (1. + self.fasterf((x - mu) / (sigma * self.sqrt(2.))))

    def uniform_pdf(self, x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
        return 1. / (hi - lo)
    
    def set_index(self, x: jnp.ndarray, index: Any, values: Any):
        x = x.at[index].set(values)
        return x
    
    def set_index_add(self, x: jnp.ndarray, index: Any, values: Any):
        x = x.at[index].add(values)
        return x
        
    def fill(self, x: jnp.ndarray, value: Any): 
        x = x.at[:].set(value)
        return x
    
    def histogram(self, x: jnp.ndarray, bins: Any, weights: jnp.ndarray) -> jnp.ndarray:
        hist, _ = jnp.histogram(x, bins=bins, weights=weights)
        return hist
    
    def bincount(self, x, weights, length) -> jnp.ndarray:
        return jnp.bincount(x, weights=weights, length=length)
    
    def reshape(self, x: jnp.ndarray, shape: Any) -> jnp.ndarray:
        return x.reshape(shape)
    
    def searchsorted(self, a, v, side = 'left'):
        return jnp.searchsorted(a, v, side)
    
    def ravel_multi_index(self, multi_index, dims):
        return jnp.ravel_multi_index(multi_index, dims, mode="clip")
    
    def clip(self, x: jnp.ndarray, low: float, high: float) -> jnp.ndarray:
        return jnp.clip(x, low, high)
    
    def linspace(self, start, stop, num):
        return jnp.linspace(start, stop, num)
    
    def where_sum(self, condition, x, y):
        return jnp.where(condition, x, y).sum()
    
    def log(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(x)
    
    def diff(self, x: jnp.ndarray, axis=0) -> jnp.ndarray:
        return jnp.diff(x, axis=axis)
    
    def allclose(self, a: jnp.ndarray, b: jnp.ndarray, atol: float = 1e-8) -> bool:
        return jnp.allclose(a, b, atol=atol)
    
    def sum(self, x: jnp.ndarray, axis: Any = None) -> jnp.ndarray:
        return jnp.sum(x, axis=axis)
    
    def tanh(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(x)

    def select(self, condition, x, y):
        return jnp.where(condition, x, y)
    
    def gammaln(self, x):
        return jax.scipy.special.gammaln(x)

    def func_and_grad(self, func):
        return jax.jit(jax.value_and_grad(func))
    
    def compile(self, func):
        return jax.jit(func)
    
    def grad(self, func):
        return jax.grad(func)
    
    def logspace(self, start, stop, num):
        return jnp.logspace(start, stop, num)
    
    def arccos(self, x):
        return jnp.arccos(x)

# Default backend instance
backend = JAXBackend()
