import random
from micrograd.variable import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):

    def __init__(self, nin, activation='tanh') -> None:
        self.w = [Value(random.uniform(-1,1)) for i in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.activation = activation

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh'
            output = activation.tanh()
        elif self.activation == 'relu'L
            output = activation.relu()
        else:
            raise ValueError(f"Unexpected activation value: {self.activation}")

        return output
    
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)}) with activation {self.activation}"

class Layer(Module):

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for i in range(nout)]

    def __call__(self, x):
        output = [n(x) for n in self.neurons]

        return output[0] if len(output)==1 else output
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self , nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
