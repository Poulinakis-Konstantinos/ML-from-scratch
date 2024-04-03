
class Tensor:
    def __init__(self, value, _children=(), _op=None) -> None:
        self.value = value
        self.grad = None
        # parameters used to track lineage of the tensor
        self.op = _op
        self.prev = _children  # previous nodes in the graph
        self.__grads__ = []

    def __add__(self, other):
        assert isinstance(
            other, Tensor), f"Cannot add types {type(self)} and {type(other)}"
        return Tensor(self.value + other.value, _children=(self, other), _op=Tensor._ops('add'))

    def __sub__(self, other):
        assert isinstance(
            other, Tensor), f"Cannot subtract type {type(other)} from type {type(self)}"
        return Tensor(self.value - other.value, _children=(self, other), _op=Tensor._ops('sub'))

    def __truediv__(self, other):
        assert isinstance(
            other, Tensor), f"Cannot divide type {type(self)} by {type(other)}"
        return Tensor(self.value / other.value, _children=(self, other), _op=Tensor._ops('truediv'))

    def __pow__(self, other):
        assert isinstance(
            other, Tensor), f"Cannot raise type {type(self)} to power of type {type(other)}"
        return Tensor(self.value ** other.value, _children=(self, other), _op=Tensor._ops('pow'))

    def __mul__(self, other):
        assert isinstance(
            other, Tensor), f"Cannot multiply types {type(self)} and {type(other)}"
        return Tensor(self.value * other.value, _children=(self, other), _op=Tensor._ops('mul'))

    def __repr__(self):
        return f"Tensor: value:{self.value}, grad:{self.grad if self.grad is not None else None}"

    @classmethod
    def _ops(cls, op: str):
        ops = {'add': cls.__add__,
               'sub': cls.__sub__,
               'truediv': cls.__truediv__,
               'pow': cls.__pow__,
               'mul': cls.__mul__}
        return ops[op]


def backward(tensor, _upstream_grad=None):
    assert isinstance(
        tensor, Tensor), f"Cannot perform backward operation on non-type {Tensor}"

    tensor.grad = Tensor(0) if _upstream_grad is None else _upstream_grad.value
    up = Tensor(1) if _upstream_grad is None else _upstream_grad
    assert isinstance(up, Tensor), f"Upstream gradient not of type {Tensor}"

    if len(tensor.prev) == 2:
        l1, l2 = __calc_grads__(tensor.op, tensor.prev[0], tensor.prev[1])
        downstream_grad1 = l1 * up
        downstream_grad2 = l2 * up
        backward(tensor.prev[0], downstream_grad1)
        backward(tensor.prev[1], downstream_grad2)
    elif len(tensor.prev) == 1:
        l1 = __calc_grad__(tensor.op, tensor.prev[0])
        downstream_grad1 = l1 * up
        backward(tensor.prev[0], downstream_grad1)


def __calc_grads__(op, _op1, _op2, _precision=0.000001):
    h = Tensor(_precision)
    local_grad_1 = (op(_op1 + h, _op2) - op(_op1, _op2)) / h
    local_grad_2 = (op(_op1, _op2 + h) - op(_op1, _op2)) / h
    return (local_grad_1, local_grad_2)


def __calc_grad__(op, _op1, _precision=0.000001):
    h = Tensor(_precision)
    local_grad = (op(_op1 + h) + op(_op1)) / h
    return local_grad


if __name__ == "__main__":
    er = 0.000001
    a = Tensor(3)
    b = Tensor(1)
    c = Tensor(-2)
    d = Tensor(2) * b
    e = a + d
    L = c*e

    backward(L)
    assert (a.grad - (-2)) < er, f"Error in gradient calculation of {a}"
    assert (b.grad - (-4)) < er, f"Error in gradient calculation of {b}"
    assert (c.grad - (5)) < er, f"Error in gradient calculation of {c}"
    assert (d.grad - (-2)) < er, f"Error in gradient calculation of {d}"
    assert (e.grad - (-2)) < er, f"Error in gradient calculation of {e}"
