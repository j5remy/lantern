"""

Tensor implementation.

- supports basic operations (add, sub, div, matmul, etc.)
- supports auto differentiation
- accelerated with CUDA

"""

import cupy as cp
import time

class Tensor:
    
    def __init__(self, data, requires_grad=False) -> None:
        if isinstance(data, (list, tuple)):
            data = cp.array(data, dtype=cp.float32)
        elif isinstance(data, cp.ndarray):
            data = data.astype(cp.float32)
        else:
            raise TypeError("Unsupported data type for Tensor initialization.")
        
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None  # Placeholder for autograd


    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.data.shape}, dtype={self.data.dtype})"

    def backward(self):
        if self.grad is None:
            self.grad = cp.ones_like(self.data)  # Initialize if first backward call
        else:
            self.grad += cp.ones_like(self.data)  # Accumulate gradients

        self._backward()  # Compute gradients
    
    # Generic function for element-wise operations
    def _elementwise_op(self, other, op, op_name):
        other = other if isinstance(other, Tensor) else Tensor(other)
        kernel = cp.ElementwiseKernel("T x, T y", "T z", f"z = x {op} y", op_name)
        out = Tensor(kernel(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                if op == "/":
                    self.grad = (cp.ones_like(self.data) / other.data) * out.grad if self.grad is None else self.grad + (cp.ones_like(self.data) / other.data) * out.grad
                else:
                    self.grad = out.grad if self.grad is None else self.grad + out.grad

            if other.requires_grad:
                if op == "/":
                    other.grad = (-self.data / (other.data ** 2)) * out.grad if other.grad is None else other.grad + (-self.data / (other.data ** 2)) * out.grad
                else:
                    other.grad = out.grad if other.grad is None else other.grad + out.grad

        out._backward = _backward
        return out

    # Basic Operations
    def __add__(self, other): return self._elementwise_op(other, "+", "add_tensors")
    def __sub__(self, other): return self._elementwise_op(other, "-", "sub_tensors")
    def __mul__(self, other): return self._elementwise_op(other, "*", "mul_tensors")
    def __truediv__(self, other): return self._elementwise_op(other, "/", "div_tensors")

    def __floordiv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(cp.floor(self.data / other.data), requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = (cp.ones_like(self.data) / other.data) * out.grad if self.grad is None else self.grad + (cp.ones_like(self.data) / other.data) * out.grad
            if other.requires_grad:
                other.grad = (-self.data / (other.data ** 2)) * out.grad if other.grad is None else other.grad + (-self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.shape[-1] == other.data.shape[0], "Matrix dimensions do not match for matmul."

        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = out.grad @ other.data.T if self.grad is None else self.grad + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = self.data.T @ out.grad if other.grad is None else other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, exponent):
        out = Tensor(self.data ** exponent, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = (exponent * self.data ** (exponent - 1)) * out.grad if self.grad is None else self.grad + (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out
    
    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis=axis, keepdims=True), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = cp.ones_like(self.data) * out.grad if self.grad is None else self.grad + cp.ones_like(self.data) * out.grad

        out._backward = _backward
        return out
    
    def mean(self, axis=None):
        out = Tensor(self.data.mean(axis=axis, keepdims=True), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = (cp.ones_like(self.data) / self.data.size) * out.grad if self.grad is None else self.grad + (cp.ones_like(self.data) / self.data.size) * out.grad

        out._backward = _backward
        return out

if __name__ == "__main__":
    tensor1 = Tensor(cp.random.rand(3, 3, 3))
    tensor2 = Tensor(cp.random.rand(3, 3, 3))
    start = time.time()
    result = tensor1 @ tensor2
    result2 = tensor1.mean()
    result3 = tensor1**2
    end = time.time()
    print(f"Elapsed time: {end - start} sec")
    print(result)
    print(result2)
    print(result3)
