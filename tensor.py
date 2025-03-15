"""

Tensor implementation.

- supports basic operations (add, sub, mul, matmul, etc.)
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
    
    def __add__(self, other: "Tensor") -> "Tensor":
        add_tensors = cp.ElementwiseKernel(
            "T x, T y",
            "T z",
            "z = x + y",
            "add_tensors"
        )

        return Tensor(add_tensors(self.data, other.data))
    
    def __sub__(self, other: "Tensor") -> "Tensor":
        pass

    def __mul__(self, other: "Tensor") -> "Tensor":
        pass

    def __matmul__(self, other: "Tensor") -> "Tensor":
        pass

    
if __name__ == "__main__":
    tensor1 = Tensor(cp.random.rand(2, 2, 2) * 100)
    tensor2 = Tensor(cp.random.rand(2, 2, 2) * 100)
    print("Tensor 1: ", tensor1)
    print("Tensor 2: ", tensor2)
    result = tensor1 + tensor2
    print(result)
