import numpy as np
import torch.nn as nn  # used only as golden truth for testing
from torch import tensor # used only as golden truth for testing
from blocks import Identity, Relu, Neuron, NaiveClassifier, Layer


def test_identity():
    identity = Identity()
    input_data = np.array([1, 2, 3])
    assert np.array_equal(identity(input_data), input_data)
    input_data = np.array([20])
    assert np.array_equal(identity(input_data), input_data)
    input_data = np.random.rand(4, 4)
    assert np.array_equal(identity(input_data), input_data)

def test_relu():
    relu = Relu()
    assert relu(5) == 5
    assert relu(-5) == 0
    assert relu(0) == 0


def test_neuron():
    n_in = 3
    neuron = Neuron(n_in, non_lin=True, w=np.ones((3)), b=0)
    assert np.array_equal(neuron.parameters(), np.array([1,1,1,0]))  # params is [w|b]

    torch_neuron = nn.Linear(n_in, 1)
    nn.init.ones_(torch_neuron.weight)
    nn.init.zeros_(torch_neuron.bias)

    x = np.array([1, 2, 3], dtype=np.float32)
    output = neuron(x)
    assert isinstance(output, np.float64)  # Ensure output is of correct type
    assert np.array_equal(output, torch_neuron(tensor(x)).detach().numpy()[0])


def test_solve_AND():
    n_in = 2
    x1 = [0, 0, 1, 1]
    x2 = [0, 1, 0, 1]
    y = np.array([0, 0, 0, 1])
    clf = NaiveClassifier()

    # define 1 possible solution
    w = np.array([1, 2])
    b = -2
    neuron1 = Neuron(n_in, non_lin=True, w=w, b=b)
    # define a 2nd possible solution
    w = np.array([1, 1])
    b = -1
    neuron2 = Neuron(n_in, non_lin=True, w=w, b=b)

    for i, x in enumerate(zip(x1, x2)):
        assert np.array_equal(clf(neuron1(np.array(x))),
                              clf(neuron2(np.array(x))), y[i])


def test_solve_OR():
    n_in = 2
    x1 = [0, 0, 1, 1]
    x2 = [0, 1, 0, 1]
    y = np.array([0, 1, 1, 1])
    clf = NaiveClassifier()

    # define 1 possible solution
    w = np.array([1, 1])
    b = -0.5
    neuron1 = Neuron(n_in, non_lin=True, w=w, b=b)
    for i, x in enumerate(zip(x1, x2)):
        assert np.array_equal(clf(neuron1(np.array(x))), y[i])


def test_Layer():
    n_in = 3
    n_units = 4
    x = np.array([2, 3, 1], dtype=np.float32)
    w_init = np.ones((n_units, n_in))
    b_init = np.zeros(n_units)

    layer = Layer(n_in, n_units, w=w_init, b=b_init)
    torch_layer = nn.Linear(n_in, n_units)
    nn.init.ones_(torch_layer.weight)
    nn.init.zeros_(torch_layer.bias)

    assert  np.array_equal(layer.parameters(), np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]])) 
    assert  np.array_equal(layer(x), torch_layer(tensor(x)).detach().numpy())
