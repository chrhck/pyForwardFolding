from typing import Dict, Any, Type, Optional, Union, Tuple
import numpy as np
from .backend import backend


class BufferManager:
    """
    Manages temporary buffers for computations.

    Args:
        datasets (Dict[str, Tuple[int, ...]]): A dictionary mapping component names to buffer lengths
    """
    def __init__(self, datasets: Dict[str, Tuple[int, ...]]):
        self.buffers = {}
        
        # Initialize buffers for each component
        for comp_name, buff_length in datasets.items():
            # Create standard buffers for each component
            self._create_buffers(comp_name, buff_length)

    def _create_buffers(self, comp_name: str, buffer_length: int) -> None:
        """
        Create standard buffers for a component.
        
        Args:
            comp_name (str): The name of the component
            hist_shape (Tuple[int, ...]): The shape of the histogram for this component
        """
        # Create empty arrays using the backend
        self.buffers[f"{comp_name}_weight"] = backend.zeros(buffer_length)
        self.buffers[f"{comp_name}_weight_sq"] = backend.zeros(buffer_length)
        self.buffers[f"{comp_name}_component"] = backend.zeros(buffer_length)

    def get_buffer(self, name: str) -> np.ndarray:
        """
        Retrieve a buffer by name.

        Args:
            name (str): The name of the buffer.

        Returns:
            np.ndarray: The buffer array.

        Raises:
            KeyError: If the buffer name does not exist.
        """
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' does not exist.")
        return self.buffers[name]

    def get_weight_buffer(self, comp_name: str) -> np.ndarray:
        """Get the weight buffer for a component."""
        return self.get_buffer(f"{comp_name}_weight")

    def get_weight_sq_buffer(self, comp_name: str) -> np.ndarray:
        """Get the weight squared buffer for a component."""
        return self.get_buffer(f"{comp_name}_weight_sq")
    
    def get_component_buffer(self, comp_name: str) -> np.ndarray:
        """Get the component buffer for a component."""
        return self.get_buffer(f"{comp_name}_component")

            
    @classmethod
    def from_datasets(cls, 
                     datasets: Dict[str, Dict[str, Any]],
                     ) -> 'BufferManager':
        """
        Create a BufferManager from datasets.
        
        Args:
            datasets: Input datasets for the analysis
            buffer_dtype: Data type for buffers
            
        Returns:
            A new BufferManager instance with properly sized buffers
        """
       
        length_mapping = {}

        for dset_name, dset in datasets.items():
            if not isinstance(dset, dict):
                raise ValueError(f"Dataset {dset_name} is not a valid dictionary.")
            
            lengths = [len(val) for val in dset.values()]
            if len(set(lengths)) != 1:
                raise ValueError(f"Dataset {dset_name} has inconsistent lengths.")
            
            # Assuming all datasets have the same length
            buffer_length = lengths[0]

            length_mapping[dset_name] = buffer_length
        
        return cls(length_mapping)

