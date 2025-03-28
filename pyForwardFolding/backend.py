from typing import Any
import jax.numpy as jnp
from math import sqrt, pi, exp


class Backend:
    """
    Abstract backend interface for numerical operations.
    """
    def array(self, data: Any) -> Any:
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

    def gauss_pdf(self, x: Any, mu: Any, sigma: Any) -> Any:
        raise NotImplementedError

    def gauss_cdf(self, x: Any, mu: Any, sigma: Any) -> Any:
        raise NotImplementedError

    def uniform_pdf(self, x: Any, lo: Any, hi: Any) -> Any:
        raise NotImplementedError


class JAXBackend(Backend):
    """
    JAX implementation of the backend interface.
    """
    def array(self, data: Any) -> jnp.ndarray:
        return jnp.array(data)

    def multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a * b

    def power(self, a: jnp.ndarray, b: float) -> jnp.ndarray:
        return a ** b

    def exp(self, a: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(a)

    def sqrt(self, a: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(a)

    def erf(self, a: jnp.ndarray) -> jnp.ndarray:
        return jnp.erf(a)

    def gauss_pdf(self, x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
        return 1. / (sigma * self.sqrt(2 * pi)) * self.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def gauss_cdf(self, x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
        return 0.5 * (1. + self.erf((x - mu) / (sigma * self.sqrt(2.))))

    def uniform_pdf(self, x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
        return 1. / (hi - lo)


# Default backend instance
backend = JAXBackend()