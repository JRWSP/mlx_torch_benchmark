import mlx.core as mx
from typing import List, Union, Tuple, Dict, Any, Optional, Callable
from mlxTensor import mlxTensor
import numpy as np

class mlxMPS:
    def __init__(self):
        self._tensors = None
        self._bond_dims = None
        self._shape = None
        self.L = None
        self._dtype = None
    
    def __createMPS__(self, tensors: List[mlxTensor]) -> 'mlxMPS':
        """
        Create an MPS from a list of tensors.
        """
        if not isinstance(tensors, list):
            raise ValueError("tensors must be a list")
        if len(tensors) == 0:
            raise ValueError("tensors list cannot be empty")
        self._tensors = []
        for tensor in tensors:
            if isinstance(tensor, mlxTensor):
                self._tensors.append(tensor)
            else:
                raise ValueError(f"Invalid tensor type: {type(tensor)}")
        self.L = len(self._tensors)
        self._dtype = self._tensors[0]._dtype if self._tensors else None
        self._get_bond_dims()
        self._get_shape()
        return self
    
    def _get_shape(self) -> List[Tuple[int]]:
        """
        Get the shape of the MPS tensors.
        """
        if self._shape is not None:
            return self._shape
        self._shape = []
        for tensor in self._tensors:
            if isinstance(tensor, mlxTensor):
                self._shape.append(tensor._shape)
            else:
                self._shape.append(tensor._shape)
        return self._shape

    def _get_bond_dims(self) -> List[int]:
        """
        Get the bond dimensions of the MPS.
        """
        if self._bond_dims is not None:
            return self._bond_dims
        bond_dims = [self._tensors[0]._shape[self._tensors[0].get_leg_index('vL')]]
        # Iterate through the tensors to get the bond dimensions
        for i in range(len(self._tensors[:-1])):
            vR = self._tensors[i]._shape[self._tensors[i].get_leg_index('vR')]
            vL = self._tensors[i+1]._shape[self._tensors[i+1].get_leg_index('vL')]
            if vR != vL:
                raise ValueError(f"Bond of tensor {i} does not match tensor {i+1}: {vR} != {vL}")
            bond_dims.append(vR)
        bond_dims.append(self._tensors[-1]._shape[self._tensors[-1].get_leg_index('vR')])
        self._bond_dims = bond_dims
        return self._bond_dims
    
    def get_tensor(self, index:int) -> mx.array:
        """
        Get the tensor at the specified index.
        """
        if index < 0 or index >= self.L:
            raise IndexError(f"Index {index} out of bounds for MPS of length {self.L}")
        return self._tensors[index]
    
    def bond_dim(self) -> List:
        """
        Get the bond dimensions of the MPS.
        """
        if self._bond_dims is None:
            self._bond_dims = self._get_bond_dims()
        return self._bond_dims
    
    def set_tensor(self, index:int, tensor:mx.array) -> None:
        """
        Set the tensor at the specified index.
        """
        if index < 0 or index >= self.L:
            raise IndexError(f"Index {index} out of bounds for MPS of length {self.L}")
        if tensor.shape != self._shape[index]:
            raise ValueError(f"Tensor shape {tensor.shape} does not match expected shape {self._shape[index]}")
        self._tensors[index] = tensor
        return self
    
    def from_product_state(self, state: List[mx.array], stream=None) -> 'mlxMPS':
        """
        Create an MPS from a product state.
        """
        tensors = [mx.reshape(tensor, shape=(1,-1,1), stream=stream) for tensor in state]
        tensors = [mlxTensor(tensor, label=['vL', 'p','vR'], dtype=tensor.dtype) for tensor in tensors]
        self.__createMPS__( tensors )
        return self
    
    def _print_MPS_backup(self) -> str:
        """
        Print the MPS tensors horizontally.
        """
        mps_str = "".join(
            [f"Tensor {i}:\n{tensor._print_label()}" for i, tensor in enumerate(self._tensors)]
        )
        return mps_str
    def _print_MPS(self) -> str:
        """
        Print the MPS tensors horizontally in the desired format with aligned columns.
        """
        L = 10 if self.L > 10 else self.L  # Limit the number of tensors to 7 for display
        max_leg_label_len = max(
            [len(tensor.get_leg_label(row)) for tensor in self._tensors[:L] for row in range(tensor._rank)]
        )
        max_shape_len = max(
            [len(str(tensor._shape[row])) for tensor in self._tensors[:L] for row in range(tensor._rank)]
        )
        column_width = max_leg_label_len + max_shape_len + 6  # Adjust column width for alignment
        header = " | ".join([f"Tn {i:<{column_width - 5}}" for i in range(L)]) + " |"
        rows = ["" for _ in range(max([tensor._rank for tensor in self._tensors[:L]]))]
        for i, tensor in enumerate(self._tensors[:L]):
            for row in range(tensor._rank):
                leg_label = tensor.get_leg_label(row)
                shape = tensor._shape[row]
                rows[row] += f"({leg_label:<{max_leg_label_len}}): {shape:<{max_shape_len}} | ".ljust(column_width)
        text = header + "\n"
        for row in rows:
            text += row + "\n"
        text += "..." if self.L > 7 else ""
        return text
        
    def __repr__(self):
        return f"mlxMPS | L={L} | \n{self._print_MPS()}"
    def __str__(self):
        return f"mlxMPS | L={L} | \n{self._print_MPS()}"
    
if __name__ == "__main__":
    d = 2 # Dimension of the local Hilbert space
    L = 13 # Length of the MPS

    tensors = [mx.array([1., 1.], dtype=mx.float32)/np.sqrt(2) for _ in range(L)]
    MPS = mlxMPS().from_product_state(tensors)
    print(MPS)
    