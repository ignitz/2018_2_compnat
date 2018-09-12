import numpy as np
import hashlib as hl
from utils import *

class Node:
    """
    doc
    """
    def __init__(self, node):
        self.children = node
    
    def eval(self, values):
        return self.children.eval(values)
    
    def walk(self):
        response = self.children.walk(self, 0)
        if type(response) is not list:
            response = [response]
        return response
    
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
        assert(type(value) is int or type(value) is float or type(value) is np.array or type(value) is np.matrix)
        self.value = float(value)

    def eval(self, values=None):
        return self.value
    
    def walk(self, parent, depth):
        return (parent, self, self.get_unique_id(), depth + 1)
    
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
    
    def __str__(self):
        if type(self.node) is Operator:
            return 'ln' + str(self.node)
        else:
            return 'ln(' + str(self.node) + ')'


def generate_operator(n_dim, depth):
    poss_operators = [0, 0, 0, 1, 1, 2, 2, 3]
    index = poss_operators[np.random.random_integers(0, len(poss_operators)-1)]
    operator = ['add','sub','mul','div'][index]
    node_left = generate_subtree(n_dim, depth-1)
    node_right = generate_subtree(n_dim, depth-1)
    # node_left = Sin(0)
    # node_right = Sin(0)
    # adjust to precedence to remove ambig...
    typeA = type(node_left)
    typeB = type(node_right)
    if (typeA is Constant) and (typeB is Constant):
        return Constant(Operator(operator, node_left, node_right).eval(0.0))

    if operator in ['add', 'sub', 'mul']:
        if typeB is Constant:
            if (typeA is not Constant):
                node_left, node_right = node_right, node_left
        elif typeB is Variable:
            if (isinstance(node_left, Function)):
                node_left, node_right = node_right, node_left
        else: # is instance of Function
            if typeB is Sin:
                if (typeA is Cos) or (typeA is Exp) or (typeA is Ln):
                    node_left, node_right = node_right, node_left
            if typeB is Cos:
                if (typeA is Exp) or (typeA is Ln):
                    node_left, node_right = node_right, node_left
            if typeB is Exp:
                if typeA is Ln:
                    node_left, node_right = node_right, node_left
    
    if operator in ['add', 'sub']:
        # try to force precedence 'parenteses' to the left
        if type(node_right) is Operator:
            if node_right.operator in ['add', 'sub']:
                operator, node_right.operator = node_right.operator, operator
                
                auxA, auxB, auxC = node_left, node_right.node_left, node_right.node_right

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
            return Operator('sub', Variable(np.random.random_integers(0, n_dim - 1)), Constant(a))
        else:
            return Operator('add', Variable(np.random.random_integers(0, n_dim - 1)), Constant(-a))

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
        return Variable(np.random.random_integers(0,n_dim - 1))
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
        values = [x/10]
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
    print(bar, ' = ', np.around(bar.eval([np.pi/2, 0]), decimals=2))

def test():

    for i in range(5):
        individual = generate_individual(2)
        print(individual.get_unique_id(), individual)
    # test_print_and_operation()

# def main():
#     test()

# if __name__ == '__main__':
#     main()