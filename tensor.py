"""

Tensor implementation.

- supports basic operations (add, matmul, div, etc)
- supports auto differentiation
- accelerated with CUDA

"""

import cupy as cp

class Tensor:
    
    def __init__(self, data, requires_grad=False) -> None:
        if isinstance(data, list) or isinstance(data, tuple):
            data = cp.array(data, dtype=cp.float32)
        elif isinstance(data, cp.ndarray):
            data = data.astype(cp.float32)
        else:
            raise TypeError("Unsupported data type for Tensor initialization.")
        
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.data.shape}, dtype={self.data.dtype})"
    
    def __add__(self, other) -> 'Tensor':
        pass
    
if __name__ == "__main__":
    data = cp.ndarray((3, 3), dtype=cp.float32)
    tensor = Tensor([1,2,3,4,5])
    print(tensor)