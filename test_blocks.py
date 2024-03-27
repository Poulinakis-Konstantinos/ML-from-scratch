import numpy as np
from blocks import Identity, Relu, Neuron, NaiveClassifier


def test_identity():
    identity = Identity()
    input_data = np.array([1, 2, 3])
    assert np.array_equal(identity(input_data), input_data)
    input_data = np.array([20])
    assert np.array_equal(identity(input_data), input_data)


def test_relu():
    relu = Relu()
    assert relu(5) == 5
    assert relu(-5) == 0
    assert relu(0) == 0


def test_neuron():
    n_in = 3
    neuron = Neuron(n_in, non_lin=True)
    x = np.array([1, 2, 3])
    output = neuron(x)
    assert isinstance(output, np.float64)  # Ensure output is of correct type


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
