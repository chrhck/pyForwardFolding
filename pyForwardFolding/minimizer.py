from typing import Callable, Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize


class Minimizer:
    """
    A wrapper for optimization routines to minimize a given objective function.

    Args:
        method (str): The optimization method to use (e.g., "L-BFGS-B", "Nelder-Mead").
        options (Dict[str, Any]): Additional options for the optimizer.
    """
    def __init__(self, method: str = "L-BFGS-B", options: Dict[str, Any] = None):
        self.method = method
        self.options = options if options is not None else {}

    def minimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        initial_guess: np.ndarray,
        bounds: Tuple[Tuple[float, float], ...] = None,
    ) -> Dict[str, Any]:
        """
        Minimize the given objective function.

        Args:
            objective_function (Callable[[np.ndarray], float]): The objective function to minimize.
            initial_guess (np.ndarray): The initial guess for the parameters.
            bounds (Tuple[Tuple[float, float], ...], optional): Bounds for the parameters. Defaults to None.

        Returns:
            Dict[str, Any]: The result of the optimization, including the optimal parameters and status.
        """
        result = minimize(
            fun=objective_function,
            x0=initial_guess,
            method=self.method,
            bounds=bounds,
            options=self.options,
        )
        return {
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "fun": result.fun,
            "x": result.x,
            "nfev": result.nfev,
            "njev": result.njev if "njev" in result else None,
        }