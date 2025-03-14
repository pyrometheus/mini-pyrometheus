

class Expression:

    def __init__(self, children):
        self.children = children

    def __add__(self, other):
        return Sum((self, other))

    def __mul__(self, other):
        return Product((self, other))

    def __truediv__(self, other):
        return Quotient(self, other)

    def __call__(self, other):
        return Call(self, other)

    def __getitem__(self, other):
        return Subscript(self, other)

    def __radd__(self, other):
        return Sum((other, self))

    def __rmul__(self, other):
        return Product((other, self))

    def __rtruediv__(self, other):
        return Quotient(other, self)


class Variable(Expression):
    mapper_method = 'map_variable'

    def __init__(self, name):
        self.name = name


class Sum(Expression):
    mapper_method = 'map_sum'

    def __add__(self, other):
        if isinstance(other, Sum):
            return Sum(self.children + other.children)
        return Sum(self.children + (other,))


class Product(Expression):
    mapper_method = 'map_product'

    def __mul__(self, other):
        if isinstance(other, Product):
            return Product(self.children + other.children)
        return Product(self.children + (other,))


class Quotient(Expression):
    mapper_method = 'map_quotient'

    def __init__(self, num, den):
        self.num = num
        self.den = den


class Call(Expression):
    mapper_method = 'map_call'

    def __init__(self, fn_name, fn_arg):
        self.fn_name = fn_name
        self.fn_arg = fn_arg


class Subscript(Expression):
    mapper_method = 'map_subscript'

    def __init__(self, ary, idx):
        self.a = ary
        self.i = idx
