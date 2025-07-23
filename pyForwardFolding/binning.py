from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from .backend import Array, backend


class AbstractBinning:
    """
    Abstract base class for binning strategies.
    """

    def __init__(self, bin_indices_dict: Optional[Dict[str, List]] = None, mask_dict: Optional[Dict[str, Array]] = None):
        if bin_indices_dict is None:
            bin_indices_dict = {}
        if mask_dict is None:
            mask_dict = {}
        self.bin_indices_dict = bin_indices_dict
        self.mask_dict = mask_dict


    @property
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
            return RelaxedBinning.construct_from(config)
        elif binning_type == "RectangularBinning":
            return RectangularBinning.construct_from(config)
        else:
            raise ValueError(f"Unknown binning type: {binning_type}")
        
    @property
    def hist_dims(self) -> Tuple[int]:
        return tuple(len(edges) - 1 for edges in self.bin_edges)
    
    @property
    def nbins(self) -> int:
        return np.prod(self.hist_dims)

    def build_histogram(
        self,
        weights: np.ndarray,
        binning_variables: Tuple[np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError


class RelaxedBinning(AbstractBinning):
    """
    Relaxed binning strategy using a tanh kernel.

    Args:
        bin_variable (str): The variable used for binning.
        bin_edges (List[float]): The edges of the bins.
        slope (float): The slope parameter for the tanh kernel.
    """
    def __init__(self, bin_variable: str, bin_edges: List[float], slope: float, mask: Any = None):
        raise NotImplementedError("RelaxedBinning is currently not implemented")
        super().__init__(None, mask)
        self.bin_variable = bin_variable
        self.bin_edges = (backend.array(bin_edges), )
        self.slope = slope

        bin_width = backend.diff(self.bin_edges[0])
        if not backend.allclose(bin_width, bin_width[0]):
            raise ValueError("Bin widths must be uniform")
        self.bin_width = bin_width[0]

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "RelaxedBinning":

        bin_edges = backend.linspace(*config["bin_edges"])
        return cls(
            bin_variable=config["bin_variable"],
            bin_edges=bin_edges,
            slope=config["slope"],
        )

    @property
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
    def __init__(
            self,
            bin_variables: Tuple[str],
            bin_edges: Tuple[List[float]],
            bin_indices_dict: Optional[Dict[str, List[Tuple[int]]]] = None,
            mask_dict: Optional[Dict[str, Any]] = None):
        super().__init__(bin_indices_dict, mask_dict)
        self.bin_variables = bin_variables
        self.bin_edges = tuple(backend.array(edges) for edges in bin_edges)

    @classmethod
    def construct_from(cls, config: Dict[str, Any]) -> "RectangularBinning":
        bin_vars_edges = config["bin_vars_edges"]
        if len(bin_vars_edges) < 1:
            raise ValueError("At least one variable and its edges must be provided.")
        bin_edges = []
        bin_variables = []
        for var, bin_type, edges in bin_vars_edges:
            bin_variables.append(var)
            if bin_type == "linear":
                bin_edges.append(backend.linspace(*edges))
            elif bin_type == "array":
                bin_edges.append(backend.array(edges))
            else:
                raise ValueError(f"Unknown binning type: {bin_type}")
        return cls(bin_variables, bin_edges)

    @property
    def required_variables(self) -> List[str]:
        return list(self.bin_variables)

    def calculate_bin_indices(self, ds_key: str, binning_variables: Tuple[np.ndarray]) -> None:
        if len(set(len(bv) for bv in binning_variables)) != 1:
            raise ValueError("All binning variables must have the same length")
       
        if ds_key not in self.bin_indices_dict:
            self.bin_indices_dict[ds_key] = []
            for bv, edges in zip(binning_variables, self.bin_edges):
                #indices = backend.searchsorted(edges, bv, side="left") - 1
                indices = backend.digitize(bv, edges) -1
                self.bin_indices_dict[ds_key].append(indices)

        if ds_key not in self.mask_dict:
            self.mask_dict[ds_key] = backend.zeros(binning_variables[0].shape, dtype=bool)
            for bv, edges in zip(binning_variables, self.bin_edges):
                self.mask_dict[ds_key] |= (bv < edges[0]) | (bv >= edges[-1])


    def clear_bin_indices(self, ds_key: str = None) -> None:
        """
        Clear the bin indices and mask for a specific dataset key, or all if ds_key is None.

        Args:
            ds_key (str, optional): The key for the dataset. If None, clears all bin indices and masks.
        """
        if ds_key is None:
            self.bin_indices_dict.clear()
            self.mask_dict.clear()
        else:
            if ds_key in self.bin_indices_dict:
                del self.bin_indices_dict[ds_key]
            if ds_key in self.mask_dict:
                del self.mask_dict[ds_key]

    def build_histogram(
        self,
        ds_key: str,
        weights: np.ndarray,
        binning_variables: Tuple[np.ndarray],
    ) -> np.ndarray:
        self.calculate_bin_indices(ds_key, binning_variables)

        indices_flat = backend.ravel_multi_index(tuple(self.bin_indices_dict[ds_key]),
                                                 self.hist_dims)
        
        # Set weight of masked samples to 0
        weights = backend.set_index(weights, self.mask_dict[ds_key], 0)

        #print(binning_variables[0][weights == 0])

        output = backend.bincount(indices_flat,
                                  weights=weights,
                                  length=self.nbins)

        return output.reshape(self.hist_dims)
