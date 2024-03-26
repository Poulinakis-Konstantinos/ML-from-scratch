'''Basic building blocks of deep learning'''
import numpy as np
from typing import List


class Identity():
    def __call__(self, x: np.array) -> np.array:
        return x
    

class Relu():
    def __init__(self) -> None:
        pass

    def __call__(self, x:np.array) -> np.array:
        return np.maximum(0, x)


class Neuron():
    '''The neural unit. Performs a weighted sum of its inputs and applies an activation function.'''
    def __init__(self, n_in:int, w:np.array=None, b:float=0, non_lin:bool = True) -> None:
        assert isinstance(n_in, int) and n_in > 0, 'Number of inputs must be a positive non-zero integer'
        
        self.w = np.random.rand(n_in) if w is None else w
        self.b = b
        self.a = Relu() if non_lin else Identity()

    def __call__(self, x:np.array) -> np.array:
        z = np.dot(self.w, x) + self.b
        return self.a(z)
        

class Layer():
    def __init__(self, n_in, n_units, **kwargs) -> None:
        self.neurons = [Neuron(n_in=n_in, **kwargs) for _ in range(n_units)]

    def __call__(self, x:np.array) -> np.Any:
        out = np.array([unit(x) for unit in self.neurons])
        return out


class MLP():
    def __init__(self, n_in:int, n_units: List(int)):
        sizes = [n_in] + n_units
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(n_units))]
    
    def __call__(self, x:np.array) -> np.Any:
        for layer in self.layers:
            x = layer(x)
        return x 
    

class Classifier():
    '''A simple two class linear classifier (negative, positive classes). Return +1 and 0'''
    def __init__(self) -> None:
        pass

    def __call__(self, x:float) -> int:
        return 1 if x>0 else 0
    

class NeuralLayer():
    '''A different approach, using the layer as the basic building block instead of the Neuron.'''

    def __init__(self, n_in:int, n_units:int, W:np.array=None, b:np.array=None, act=None) -> None:
        assert isinstance(n_in, int) and n_in > 0, 'Number of inputs must be a positive non-zero integer'
        assert isinstance(n_units, int) and n_units > 0, 'Number of neurons in a layer must be a positive non-zero integer'
        if W is not None:
            assert W.shape == (n_in, n_units), f"Invalid shape for layer weight's matrix W: {W.shape}, expected {(n_in, n_units)}"
        if b is not None:
            assert b.shape == (n_units,), f"Invalid shape for layer bias b: {b.shape}, expected {(n_units,)}"

        self.n_in = n_in
        self.units = n_units
        self.W = np.random.rand(n_in, n_units) if W is None else W
        self.b = np.ones(n_units) if b is None else b
        self.act = Relu() if act is None else act

    def __call__(self, x:np.array) -> np.Any:
        assert x.shape == (self.n_in, ), f"Invalid shape for input x: {x.shape}, expected {(self.n_in,)}"
        return self.act(np.add(np.dot(x, self.w), self.b))


