import numbers
import numpy as np


# {{{ Graph Walker

class AutodiffWalker:

    def compute_gradient(self, ary):
        prec_grad = np.ones_like(ary.values)
        self.gradients = {}
        self.rec(ary, prec_grad)
        return self.gradients

    def rec(self, ary, prec_grad):
        if isinstance(ary, AutodiffVariable):
            if ary.name in self.gradients:
                self.gradients[ary.name] += prec_grad
            else:
                self.gradients[ary.name] = prec_grad
        else:
            self.propagate_gradients(ary, prec_grad)

    def propagate_gradients(self, ary, prec_grad):
        for c, g in zip(ary.children, ary.grad_fn(prec_grad)):
            if isinstance(c, AutodiffArray):
                self.rec(c, g)

# }}}


# {{{ Loopy Instruction Mapper

class LoopyMapper:

    def rec(self, ary, *args):
        import numbers
        if isinstance(ary, numbers.Number):
            return f'{ary}'
        if args:
            return getattr(self, ary.mapper_method)(ary, *args)
        else:
            return getattr(self, ary.mapper_method)(ary, None)

    def map_sum(self, ary, prec):
        return self.parenthesize(
            ' + '.join([
                self.rec(c, prec) for c in ary.children
            ]), self.prec['sum'], prec)

    def map_product(self, ary, prec):
        return self.parenthesize(
            ' * '.join([
                self.rec(c, prec) for c in ary.children
            ]), self.prec['mul'], prec)

    def map_quotient(self, ary, prec):
        return self.parenthesize(
            ' / '.join([
                self.rec(ary.c) for c in ary.children
            ]), self.prec['div'], prec)

    def map_variable(self, ary, *args):
        dim = len(ary.shape)
        idx = tuple(f'i{i}' for i in range(dim))
        return ary.expr.name + '[{:s}]'.format(', '.join([i for i in idx]))

# }}}


# {{{ Arrays

class AutodiffArray:

    def __init__(self, values: list, children, name=None):
        self.values = np.array(values, dtype=np.float64)
        self.children = children
        self.name = name

    @property
    def shape(self,):
        return self.values.shape

    def __add__(self, other):
        if isinstance(other, AutodiffArray):
            return AutodiffSum(
                self.values + other.values,
                children=[self, other]
            )
        elif isinstance(other, numbers.Number):
            return AutodiffSum(
                self.values + other,
                children=[self, other]
            )
        else:
            raise ValueError

    def __mul__(self, other):
        if isinstance(other, AutodiffArray):
            return AutodiffProduct(
                self.values * other.values,
                children=[self, other]
            )
        elif isinstance(other, numbers.Number):
            return AutodiffProduct(
                self.values * other,
                children=[self, other]
            )
        else:
            raise ValueError

    def __getitem__(self, idx):
        return AutodiffSubscript(
            values=self.values,
            children=[self,],
            idx=idx
        )

    def __radd__(self, other):
        if isinstance(other, AutodiffArray):
            return AutodiffSum(
                self.values + other.values,
                children=[self, other]
            )
        elif isinstance(other, numbers.Number):
            return AutodiffSum(
                self.values + other,
                children=[self, other]
            )
        else:
            raise ValueError

    def __rtruediv__(self, other):
        if isinstance(other, AutodiffArray):
            return AutodiffRevQuotient(
                other.values / self.values,
                children=[self, other]
            )
        elif isinstance(other, numbers.Number):
            return AutodiffRevQuotient(
                other / self.values,
                children=[self, other]
            )
        else:
            raise ValueError

    def __rmul__(self, other):
        if isinstance(other, AutodiffArray):
            return AutodiffProduct(
                self.values * other.values,
                children=[self, other]
            )
        elif isinstance(other, numbers.Number):
            return AutodiffProduct(
                self.values * other,
                children=[self, other]
            )
        else:
            raise ValueError

    def zero_grads(self,):
        self.grad_values = np.zeros_like(self.values)

    def gradient(self,):
        ad_walker = AutodiffWalker()
        return ad_walker.compute_gradient(self)


class AutodiffVariable(AutodiffArray):

    def __init__(self, values, name):
        self.values = values
        self.name = name
        self.children = []

    def grad_fn(self, grad):
        return np.zeros_like(grad)


class AutodiffSum(AutodiffArray):

    def grad_fn(self, grad):
        if isinstance(self.children[1], AutodiffArray):
            return (grad, grad)
        else:
            return (grad,)


class AutodiffProduct(AutodiffArray):

    def grad_fn(self, grad):
        if isinstance(self.children[1], AutodiffArray):
            return (
                grad * self.children[1].values,
                grad * self.children[0].values
            )
        else:
            return (
                grad * self.children[1],
            )


class AutodiffRevQuotient(AutodiffArray):

    def grad_fn(self, grad):
        if isinstance(self.children[1], AutodiffArray):
            return (
                grad / self.children[0].values,
                -grad * self.children[1].values / self.children[0].values ** 2
            )
        else:
            return (
                -grad * self.children[1] / self.children[0].values ** 2,
            )


class AutodiffSubscript(AutodiffArray):

    def __init__(self, values, children, idx):
        self.values = values[idx]
        self.children = children
        self.idx = idx
        self.p_shape = values.shape

    def grad_fn(self, grad):
        out_grad = np.zeros(self.p_shape)
        out_grad[self.idx] = grad
        return (out_grad,)


# }}}


# {{{ Math

def exp(ary: AutodiffArray):
    def grad_fn(grad):
        return (grad * np.exp(ary.values),)
    new_ary = AutodiffArray(
        np.exp(ary.values),
        children=[ary]
    )
    new_ary.grad_fn = grad_fn
    return new_ary


def log(ary: AutodiffArray):
    def grad_fn(grad):
        return (grad / ary.values,)
    new_ary = AutodiffArray(
        np.log(ary.values),
        children=[ary]
    )
    new_ary.grad_fn = grad_fn
    return new_ary

# }}}
