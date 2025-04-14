from typing import List, Tuple, Dict, Any, Type
import numpy as np
from .backend import backend


class AbstractBinning:
    """
    Abstract base class for binning strategies.
    """
    def required_variables(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def construct_from(cls: Type["AbstractBinning"], config: Dict[str, Any]) -> "AbstractBinning":
        """
        Construct a binning object from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            AbstractBinning: An instance of a subclass of AbstractBinning.
        """
        binning_type = config.get("type")
        if binning_type == "CustomBinning":
            return CustomBinning(bin_indices=config["bin_indices"])
        elif binning_type == "RelaxedBinning":
            return RelaxedBinning(
                bin_variable=config["bin_variable"],
                bin_edges=config["bin_edges"],
                kernel_buffer=np.zeros(len(config["bin_edges"]) - 1),
                slope=config["slope"],
            )
        elif binning_type == "RectangularBinning":
            return RectangularBinning.from_pairs(config["bin_vars_edges"])
        else:
            raise ValueError(f"Unknown binning type: {binning_type}")
        
    @property
    def hist_dims(self) -> Tuple[int]:
        return tuple(len(edges) - 1 for edges in self.bin_edges)
    
    @property
    def nbins(self) -> int:
        return np.prod(self.hist_dims)


class CustomBinning(AbstractBinning):
    """
    Custom binning strategy.

    Args:
        bin_indices (List[int]): Indices for the bins.
    """
    def __init__(self, bin_indices: List[int]):
        super().__init__()
        self.bin_indices = bin_indices

    def required_variables(self) -> List[str]:
        return []


class RelaxedBinning(AbstractBinning):
    """
    Relaxed binning strategy using a tanh kernel.

    Args:
        bin_variable (str): The variable used for binning.
        bin_edges (List[float]): The edges of the bins.
        slope (float): The slope parameter for the tanh kernel.
    """
    def __init__(self, bin_variable: str, bin_edges: List[float], slope: float):
        super().__init__()
        self.bin_variable = bin_variable
        self.bin_edges = (backend.array(bin_edges), )
        self.slope = slope

        bin_width = backend.diff(self.bin_edges[0])
        if not backend.allclose(bin_width, bin_width[0]):
            raise ValueError("Bin widths must be uniform")
        self.bin_width = bin_width[0]

    def required_variables(self) -> List[str]:
        return [self.bin_variable]

    def _tanh_bin_kernel(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return 0.5 * (1 + backend.tanh((x - a) / self.slope) * backend.tanh(-(x - b) / self.slope))

    def _tanh_bin_kernel_norm(self, bin_width: float) -> float:
        return 1 / backend.tanh(bin_width / self.slope)

    def build_histogram(
        self,
        weights: np.ndarray,
        binning_variables: Tuple[np.ndarray],
    ) -> np.ndarray:
        if len(binning_variables) != 1:
            raise ValueError("RelaxedBinning only supports one binning variable")

        data = binning_variables[0]


        output = backend.zeros(self.hist_dims)

        lower_edges = self.bin_edges[0][:-1]
        upper_edges = self.bin_edges[0][1:]

        for i, (le, ue) in enumerate(zip(lower_edges, upper_edges)):
            output = backend.set_index(output, i, backend.sum(self._tanh_bin_kernel(data, le, ue) * weights))

        output /= self._tanh_bin_kernel_norm(self.bin_width)
        return output


class RectangularBinning(AbstractBinning):
    """
    Rectangular binning strategy for multi-dimensional data.

    Args:
        bin_variables (Tuple[str]): The variables used for binning.
        bin_edges (Tuple[List[float]]): The edges of the bins for each variable.
        bin_indices (List[Tuple[int]]): Precomputed bin indices.
    """
    def __init__(self, bin_variables: Tuple[str], bin_edges: Tuple[List[float]], bin_indices: List[Tuple[int]] = None):
        super().__init__()
        self.bin_variables = bin_variables
        self.bin_edges = tuple(backend.array(edges) for edges in bin_edges)
        self.bin_indices = bin_indices if bin_indices is not None else []

    @classmethod
    def from_pairs(cls, bin_vars_edges: List[Tuple[str, List[float]]]) -> "RectangularBinning":
        bin_edges = tuple(backend.linspace(*edges) for _, edges in bin_vars_edges)
        bin_variables = tuple(var for var, _ in bin_vars_edges)
        return cls(bin_variables, bin_edges)

    def required_variables(self) -> List[str]:
        return list(self.bin_variables)

    def calculate_bin_indices(self, binning_variables: Tuple[np.ndarray]) -> None:
        if len(set(len(bv) for bv in binning_variables)) != 1:
            raise ValueError("All binning variables must have the same length")

        if not self.bin_indices:
            self.bin_indices = []
            for bv, edges in zip(binning_variables, self.bin_edges):
                indices = backend.searchsorted(edges, bv, side="left") - 1
                self.bin_indices.append(indices)

    def build_histogram(
        self,
        weights: np.ndarray,
        binning_variables: Tuple[np.ndarray],
    ) -> np.ndarray:
        if not self.bin_indices:
            self.calculate_bin_indices(binning_variables)

        indices_flat = backend.ravel_multi_index(tuple(self.bin_indices), self.hist_dims)
        output = backend.bincount(indices_flat, weights=weights, length=self.nbins)
        output = backend.reshape(output, self.hist_dims)

        return output