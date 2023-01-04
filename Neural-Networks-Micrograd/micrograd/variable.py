# Rewriting Value with gradient calculation
import numpy as np
import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = _children
        self.grad = 0.0
        self._backward = lambda: None
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data}, label={self.label}, grad={self.grad})"
        
    def __add__(self, other):
        if np.isscalar(other):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad  # Local gradient is 1 for addition, which is multiplied by the incoming gradient from the output
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        return self + (-1*other)

    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __mul__(self, other):
        if np.isscalar(other):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data*out.grad # Local derivative of multiplication is just the variable which is being multiplied to.
            other.grad += self.data*out.grad # Local derivative then multiplied with the incoming gradient from the output
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        if np.isscalar(other):
            other = Value(other)
        return self*other**-1

    def __rtruediv__(self, other):
        return other*self**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad = (other*self.data**(other-1))*out.grad   # Local derivative of power function chained with output grad
        out._backward = _backward
        return out

    def exp(self):
        n = self.data
        exp_value = math.exp(n)
        out = Value(exp_value, (self, ), 'exp')

        def __backward():
            self.grad = out.data*out.grad   # Local gradient of exponentiation is itself
        out._backward = __backward

        return out

    def tanh(self):
        n = self.data
        tanh_value = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(tanh_value, (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2)*out.grad
        out._backward = _backward

        return out

    def relu(self):
        n = self.data
        relu_value = n if n > 0 else 0
        out = Value(relu_value, (self, ), 'relu')

        def _backward():
            self.grad += (1.0*out.grad if n > 0 else 0) # Relu has zero local gradient when n <=0. Otherwise it is y=x function with local gradient 1.0
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        def build_topo(v, visited):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child, visited)
                topo.append(v)
        build_topo(self, set())
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
