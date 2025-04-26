import mlx
import mlx.core as mx
import numpy as np
from typing import List, Union, Tuple, Dict, Any, Optional, Callable

import numpy as np

def random_unitary_matrix(dim: int) -> mx.array:
    """
    Generate a random unitary matrix of size (dim x dim).
    """
    # Create a random complex matrix
    random_matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    
    # Perform QR decomposition
    q, r = np.linalg.qr(random_matrix)
    
    # Ensure the matrix is unitary by normalizing with the diagonal of R
    d = np.diag(r)
    q = q * (d / np.abs(d))
    
    return mx.array(q, dtype=mx.complex64)

class mlxTensor:
    def __init__(self, data, label:List=None, dtype=None):
        self._data = mx.array(data, dtype=dtype)
        self._shape:List[int] = None
        self._rank:int = None
        self._label:List[str|int] = None
        self._dtype = self._data.dtype
        self.__set_shape__()
        self.__set_label__(label)
        
    def __set_label__(self, label:List=None):
        if label is not None:
            if len(label) != self._rank:
                raise ValueError(f"Label length {len(label)} does not match tensor rank {self._rank}")
            self._label = label
        else:
            self._label = [str(label) for label in range(self._rank)]
        return self
            
    def __set_shape__(self):
        if self._data is not None:
            self._shape = self._data.shape
            self._rank = len(self._shape)
            return self
        else:
            raise ValueError("Data is None, cannot set shape")
        
    def transpose(self, axes:List[int|str]=None, stream=None) -> None:
        if axes is None:
            axes = list(reversed(range(self._rank)))
        if isinstance(axes, list):
            axes = [self._label.index(axis) if isinstance(axis, str) else axis for axis in axes]
        else:
            raise ValueError("axes must be a list of int or str")
        if axes == list(range(self._rank)):
            return self  # nothing to do
        if all(isinstance(axis, int) for axis in axes):
            self._data = mx.transpose(self._data, axes, stream=stream)
            self.__set_label__([self._label[i] for i in axes])
            self.__set_shape__()
            return self
        else:
            raise ValueError("All elements in axes must be either int or str")

    def swapaxes(self, axis1:int|str, axis2:int|str, stream=None) -> None:
        axis1 = self._label.index(axis1) if isinstance(axis1, str) else axis1
        axis2 = self._label.index(axis2) if isinstance(axis2, str) else axis2
        if axis1 == axis2:
            return self
        if axis1 < 0 or axis1 > self._rank-1 or axis2 < 0 or axis2 > self._rank-1:
            raise ValueError(f"axis1={axis1} or axis2={axis2} are out of bounds for tensor of rank {self._rank}")
        self._data = mx.swapaxes(self._data, axis1, axis2, stream=stream)
        label = self._label
        label[axis1], label[axis2] = label[axis2], label[axis1]
        self.__set_label__(label)
        self.__set_shape__()
        return self
    
    def combine_legs(self, combine_legs:List[List[int]], new_axes=None, stream=None) -> None:
        pass
        return self
    
    def split_legs(self, axes=None, stream=None):
        pass
        return self
    
    def get_leg_index(self, label:str) -> int:
        """Return index of leg label"""
        if label not in self._label:
            raise ValueError(f"Label {label} not found in tensor labels {self._label}")
        return self._label.index(label)
    
    def get_leg_label(self, index:int) -> str:
        """Return label of leg index"""
        if index < 0 or index >= self._rank:
            raise ValueError(f"Index {index} out of bounds for tensor of rank {self._rank}")
        return self._label[index]
    
    def to_array(self) -> mx.array:
        """Return the data as mlx array"""
        return self._data

    def conj(self):
        """Conjugate the tensor"""
        self._data = mx.conj(self._data)
        new_label = [l+"*" if l[-1] != "*" else l[:-1] for l in self._label]
        self.__set_label__(new_label)
        return self
    
    def set_label(self, label:List):
        """Set label from list of strings"""
        self.__set_label__(label)
        return self
        
    def _print_label(self)->str:
        label_str = ""
        for label, dim in zip(self._label, self._shape):
            text_space = " "*(10-len(f"({label})"))
            label_str += f"({label}){text_space}: {dim}\n"
        return label_str
    def __repr__(self):
        return f"mlxTensor | type={self._data.dtype}\n{self._print_label()}"
    def __str__(self):
        return f"mlxTensor | type={self._data.dtype}\n{self._print_label()}"

if __name__ == "__main__":
    # Create a random tensor
    shape = (2,3,4,5)
    dtype = mx.float32
    data = mx.random.normal(shape=shape, dtype=dtype) + 1j*mx.random.normal(shape=shape, dtype=dtype)
    #data = random_unitary_matrix(3)
    label = ["vL", "p", "p*", "vR"]
    stream = None
    tensor = mlxTensor(data, label=label, dtype=dtype)
    # Print the tensor
    
    print(data)
    print(tensor)
    print(tensor.swapaxes("p", "p*"))
    print(tensor.swapaxes(1, 2))
    