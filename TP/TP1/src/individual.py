"""Individual."""
import hashlib as hl

import numpy as np

from utils import print_blue


class Node:
    """doc."""

    def __init__(self, node):
        self.children = node

    def calc_fitness(self, data):
        """Calculation of fitness by NRMSE."""
        y_mean = data[:, -1].mean()
        normalize = data[:, -1] - y_mean
        normalize = np.sum(normalize)
        real_diff = list()
        for d in data:
            eval_value = self.eval(d[:, :-1])
            real_value = d[:, -1].item(0)
            real_diff.append((eval_value - real_value)**2)
        if float('inf') in real_diff:
            return float('inf')
        else:
            return np.sqrt(np.sum(real_diff))
            # return np.sqrt(np.sum(real_diff) / normalize)

    def eval(self, values):
        return self.children.eval(values)

    def walk(self):
        response = self.children.walk(self, 0)
        if type(response) is not list:
            response = [response]
        return response

    def copy(self):
        return Node(self.children.copy())

    def __str__(self):
        return self.children.__str__()

    def get_unique_id(self):
        """
        Return shasum to detect identical individual
        """
        string = self.__str__()
        return hl.sha256(string.encode('utf-8')).hexdigest()


class Constant(Node):
    """
    doc
    """
    value = 0.0

    def __init__(self, value):
        self.value = float(value)

    def eval(self, values=None):
        return self.value

    def walk(self, parent, depth):
        return (parent, self, self.get_unique_id(), depth + 1)

    def copy(self):
        return Constant(self.value)

    def __str__(self):
        return str(np.around(self.value, decimals=2))


class Operator(Node):
    """
    doc
    """

    def __init__(self, operator, node_left, node_right):
        assert(type(operator) is str)
        assert(isinstance(node_left, Node))
        assert(isinstance(node_right, Node))
        if operator not in ['add', 'sub', 'mul', 'div']:
            operator = {
                '+': 'add',
                '-': 'sub',
                '*': 'mul',
                'x': 'mul',
                '/': 'div',
            }[operator]
        self.operator = operator
        self.node_left = node_left
        self.node_right = node_right

    def handle_div(self, values):
        """
        return max of dataset if div 0... how to get this max? global?

        """
        if self.node_right.eval(values) == 0.0:
            return float('inf')
        else:
            return self.node_left.eval(values) / self.node_right.eval(values)

    def eval(self, values):
        a = self.node_left.eval(values)
        b = self.node_right.eval(values)
        if np.isnan(a) or np.isnan(b) or a == np.inf or b == np.inf:
            return float('inf')
        if self.operator == 'add':
            return a + b
        elif self.operator == 'sub':
            return a - b
        elif self.operator == 'mul':
            return a * b
        elif self.operator == 'div':
            divisor = b
            if divisor == 0.0:
                return float('inf')
            return a / divisor
        else:
            raise 'operator invalid'

    def walk(self, parent, depth):
        a = self.node_left.walk(self, depth + 1)
        if type(a) is not list:
            response = [a]
        else:
            response = a
        b = self.node_right.walk(self, depth + 1)
        if type(b) is not list:
            response += [b]
        else:
            response += b
        response += [(parent, self, self.get_unique_id(), depth + 1)]
        return response

    def copy(self):
        return Operator(self.operator, self.node_left.copy(),
                        self.node_right.copy())

    def get_operator_char(self):
        return {
            'add': '+',
            'sub': '-',
            'mul': 'x',
            'div': '/'
        }[self.operator]

    def __str__(self):
        return '(' + self.node_left.__str__() + \
            ' ' + self.get_operator_char() + ' ' + \
            self.node_right.__str__() + ')'


class Variable(Node):
    """
    doc
    """
    def __init__(self, dim):
        self.dim = dim

    def get_dim(self):
        return self.dim

    def eval(self, values):
        return values.item(self.dim)

    def walk(self, parent, depth):
        return (parent, self, self.get_unique_id(), depth + 1)

    def copy(self):
        return Variable(self.dim)

    def __str__(self):
        return 'X' + str(self.dim)


class Function(Node):
    """
    Abstract class
    f(x)
    """
    def __init__(self, node):
        self.node = node

    def walk(self, parent, depth):
        a = self.node.walk(self, depth + 1)
        if type(a) is not list:
            response = [a]
        else:
            response = a
        return response + [(parent, self, self.get_unique_id(), depth + 1)]


class Sin(Function):
    """
    sin(x_i)
    """
    def eval(self, values):
        return np.sin(self.node.eval(values))

    def copy(self):
        return Sin(self.node.copy())

    def __str__(self):
        if type(self.node) is Operator:
            return 'sin' + str(self.node)
        else:
            return 'sin(' + str(self.node) + ')'


class Cos(Function):
    """
    cos(x_i)
    """
    def eval(self, values):
        return np.cos(self.node.eval(values))

    def copy(self):
        return Cos(self.node.copy())

    def __str__(self):
        if type(self.node) is Operator:
            return 'cos' + str(self.node)
        else:
            return 'cos(' + str(self.node) + ')'


class Exp(Function):
    """
    e^(x_i)
    """
    def eval(self, values):
        return np.exp(self.node.eval(values))

    def copy(self):
        return Exp(self.node.copy())

    def __str__(self):
        if type(self.node) is Operator:
            return 'exp' + str(self.node)
        else:
            return 'exp(' + str(self.node) + ')'


class Ln(Function):
    """
    ln(x_i)
    if x <= 0 then return inf
    """
    def eval(self, values):
        eval_value = self.node.eval(values)
        if eval_value <= 0.0:
            return float('inf')
        else:
            return np.log(eval_value)

    def copy(self):
        return Ln(self.node.copy())

    def __str__(self):
        if type(self.node) is Operator:
            return 'ln' + str(self.node)
        else:
            return 'ln(' + str(self.node) + ')'


def generate_operator(n_dim, depth):
    poss_operators = [0, 0, 0, 1, 1, 2, 2, 3]
    random_int = np.random.random_integers(0, len(poss_operators) - 1)
    index = poss_operators[random_int]
    operator = ['add', 'sub', 'mul', 'div'][index]
    node_left = generate_subtree(n_dim, depth - 1)
    node_right = generate_subtree(n_dim, depth - 1)
    # node_left = Sin(0)
    # node_right = Sin(0)
    # adjust to precedence to remove ambig...
    type_a = type(node_left)
    type_b = type(node_right)
    if (type_a is Constant) and (type_b is Constant):
        return Constant(Operator(operator, node_left, node_right).eval(0.0))

    if operator in ['add', 'sub', 'mul']:
        if type_b is Constant:
            if (type_a is not Constant):
                node_left, node_right = node_right, node_left
        elif type_b is Variable:
            if (isinstance(node_left, Function)):
                node_left, node_right = node_right, node_left
        # is instance of Function
        else:
            if type_b is Sin:
                if (type_a is Cos) or (type_a is Exp) or (type_a is Ln):
                    node_left, node_right = node_right, node_left
            if type_b is Cos:
                if (type_a is Exp) or (type_a is Ln):
                    node_left, node_right = node_right, node_left
            if type_b is Exp:
                if type_a is Ln:
                    node_left, node_right = node_right, node_left

    if operator in ['add', 'sub']:
        # try to force precedence 'parenteses' to the left
        if type(node_right) is Operator:
            if node_right.operator in ['add', 'sub']:
                operator, node_right.operator = node_right.operator, operator

                auxA, auxB, auxC = (node_left,
                                    node_right.node_left,
                                    node_right.node_right
                                    )

                node_left = node_right
                node_right = auxC
                node_left.node_left = auxA
                node_left.node_right = auxB

        # factor to 2 times a node
        if type(node_left) is type(node_right):
            if node_left.get_unique_id() == node_right.get_unique_id():
                if operator == 'add':
                    node_left = Constant(2)
                else:
                    node_left = Constant(-2)
                operator = 'mul'

    if operator == 'div':
        if node_left.get_unique_id() == node_right.get_unique_id():
            return Constant(1)

        if type(node_right) is Constant:
            if node_right.eval() == 1.0:
                return node_left

    return Operator(operator, node_left, node_right)


def generate_terminal(n_dim):
    choose = np.random.random_integers(0, 5)
    if choose == 1:
        return Variable(np.random.random_integers(0, n_dim - 1))
    else:
        a = np.random.normal(scale=10)
        if a > 0:
            return Operator(
                'sub', Variable(np.random.random_integers(0, n_dim - 1)),
                Constant(a))
        else:
            return Operator(
                'add', Variable(np.random.random_integers(0, n_dim - 1)),
                Constant(-a))


def generate_subtree(n_dim, depth):
    """
    Generate a random math expression
    """

    if depth <= 0:
        return generate_terminal(n_dim)

    choose = np.random.random_integers(0, depth + 8)

    if 0 <= choose <= depth:
        return generate_operator(n_dim, depth - 1)
    elif depth < choose <= depth + 3:
        return Variable(np.random.random_integers(0, n_dim - 1))
    elif choose == depth + 4:
        return Constant(np.random.normal(scale=10))
    elif choose == depth + 5:
        return Sin(generate_terminal(n_dim))
    elif choose == depth + 6:
        return Cos(generate_terminal(n_dim))
    elif choose == depth + 7:
        return Exp(generate_terminal(n_dim))
    elif choose == depth + 8:
        return Ln(generate_terminal(n_dim))


def generate_individual(n_dim, depth=7):
    return Node(generate_subtree(n_dim, depth))


def test_basic_operations():
    x1 = Variable(0)
    x2 = Variable(1)
    values = np.array([1.0, 0.0])
    print(x1.eval(values), x2.eval(values))
    node = Operator('add', x1, x2)
    print(node.eval(values))
    node = Operator('sub', x1, x2)
    print(node.eval(values))
    node = Operator('mul', x1, x2)
    print(node.eval(values))
    node = Operator('div', x1, x2)
    print(node.eval(values))


def test_functions():
    sin = Sin(0)
    cos = Cos(0)
    exp = Exp(0)
    ln = Ln(0)
    for x in range(-10, 105, 5):
        values = [x / 10]
        print_blue("Sin(" + str(values[0]) + ") = " + str(sin.eval(values)))
        print_blue("Cos(" + str(values[0]) + ") = " + str(cos.eval(values)))
        print_blue("E(" + str(values[0]) + ") = " + str(exp.eval(values)))
        print_blue("LN(" + str(values[0]) + ") = " + str(ln.eval(values)))


def test_print_and_operation():
    a = Constant(1)
    b = Constant(2)
    foo = Operator('add', a, b)
    print(foo)
    a = Ln(0)
    b = Cos(1)
    bar = Operator('+', a, b)
    print(bar, ' = ', np.around(bar.eval([np.pi / 2, 0]), decimals=2))


def test():
    inds = []
    inds.append(generate_individual(1))
    inds.append(generate_individual(1))
    inds.append(inds[1].copy())
    for ind in inds:
        print(ind.get_unique_id(), ind)


def main():
    test()

if __name__ == '__main__':
    main()
