import numpy as np

class Node:
    def __init__(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError

class Constant(Node):
    value = 0.0

    def __init__(self, value):
        assert(type(value) is float or np.array)
        self.value = value

    def eval(self, values):
        return self.value

class Operator(Node):
    operator = None
    node_left = None
    node_right = None

    def __init__(self, operator, node_left, node_right):
        assert(type(operator) is str)
        assert(isinstance(node_left, Node))
        assert(isinstance(node_right, Node))
        self.operator = operator
        self.node_left = node_left
        self.node_right = node_right
        print(self.node_left.eval(np.array([1.0,2.0])), self.node_right.eval(np.array([1.0,2.0])))
    
    def handle_div(self, values):
        """
        return max of dataset if div 0... how to get this max? global?

        """
        if self.node_right.eval(values) == 0.0:
            return 100000000000000.0
        else:
            return self.node_left.eval(values) / self.node_right.eval(values)
    
    def eval(self, values):
        return {
            'add': self.node_left.eval(values) + self.node_right.eval(values),
            'sub': self.node_left.eval(values) - self.node_right.eval(values),
            'mul': self.node_left.eval(values) * self.node_right.eval(values),
            'div': self.handle_div(values)
        }[self.operator]

"""
Index?
multi variable?
"""

class Variable(Node):
    def __init__(self, dim):
        self.dim = dim
    
    def get_dim(self):
        return self.dim

    def eval(self, values):
        return values[self.dim]


class Function(Node):
    def __init__(self):
        raise NotImplementedError
    
class Sin(Function):



def main():
    x1 = Variable(0)
    x2 = Variable(1)
    values = np.array([1.0, 2.0])
    print(x1.eval(values), x2.eval(values))
    node = Operator('add', x1, x2)
    print(node.eval(values))
    node = Operator('sub', x1, x2)
    print(node.eval(values))
    node = Operator('mul', x1, x2)
    print(node.eval(values))
    node = Operator('div', x1, x2)
    print(node.eval(values))

if __name__ == '__main__':
    main()
