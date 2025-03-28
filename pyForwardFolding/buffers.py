from typing import Dict, Any, Type
import numpy as np


class BufferManager:
    """
    Manages temporary buffers for computations.

    Args:
        buffer_shapes (Dict[str, tuple]): A dictionary mapping buffer names to their shapes.
        buffer_dtype (Type): The data type of the buffers (e.g., np.float32, np.float64).
    """
    def __init__(self, buffer_shapes: Dict[str, tuple], buffer_dtype: Type = np.float64):
        self.buffers = {
            name: np.zeros(shape, dtype=buffer_dtype) for name, shape in buffer_shapes.items()
        }

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

    def reset_buffer(self, name: str) -> None:
        """
        Reset a buffer to zeros.

        Args:
            name (str): The name of the buffer.

        Raises:
            KeyError: If the buffer name does not exist.
        """
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' does not exist.")
        self.buffers[name].fill(0)

    def reset_all_buffers(self) -> None:
        """
        Reset all buffers to zeros.
        """
        for buffer in self.buffers.values():
            buffer.fill(0)