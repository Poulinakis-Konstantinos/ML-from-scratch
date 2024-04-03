
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


