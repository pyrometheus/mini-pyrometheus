import numpy as np
import pycuda.gpuarray as gpuarray
from minipyro.symbolic import (
    Variable, Expression, Call, Sum,
    Product, Quotient, Subscript
)
from minipyro.pyro_np.loopy import assemble_cuda


# {{{

class LazyArray:

    def __add__(self, other):
        if isinstance(other, LazyArray):
            return broadcast_binary_op(self, other, Sum)
        else:
            return ArrayExpression(
                expr=Sum((self, other)),
                shape=self.shape
            )

    def __mul__(self, other):
        if isinstance(other, LazyArray):
            return broadcast_binary_op(self, other, Product)
        else:
            return ArrayExpression(
                expr=Product((self, other)),
                shape=self.shape
            )

    def __getitem__(self, idx):
        return ArrayExpression(
            expr=Subscript(self, idx),
            shape=self.shape[1:]
        )

    def __radd__(self, other):
        if isinstance(other, LazyArray):
            return broadcast_binary_op(self, other, Sum)
        else:
            return ArrayExpression(
                expr=Sum((other, self)),
                shape=self.shape
            )

    def __rmul__(self, other):
        if isinstance(other, LazyArray):
            return broadcast_binary_op(self, other, Product)
        else:
            return ArrayExpression(
                expr=Product((other, self)),
                shape=self.shape
            )

    def __rtruediv__(self, other):
        if isinstance(other, LazyArray):
            return broadcast_binary_op(other, self, Quotient)
        else:
            return ArrayExpression(
                expr=Quotient(other, self),
                shape=self.shape
            )


class Placeholder(LazyArray):

    def __init__(self, name, shape):
        self.expr = Variable(name)
        self.shape = shape

    @property
    def name(self):
        return self.expr.name


class ArrayExpression(LazyArray):

    def __init__(self, expr, shape):
        self.expr = expr
        self.shape = shape
        self.cuda_prg = None

    def compile(self, knl_name, wg_size):
        self.wg_size = wg_size
        self.cuda_prg = assemble_cuda(self, knl_name)

    def evaluate(self, *np_data):
        assert self.cuda_prg is not None

        ws = self.wg_size
        dim = len(self.shape)
        shape = self.shape + (1,) if dim == 2 else self.shape
        print(shape)
        grid = tuple(
            (s + ws - 1)//ws for s in shape
        )
        block = (ws, ws, 1) if dim == 2 else (ws, ws, ws)

        dev_data = [gpuarray.to_gpu(a) for a in np_data]
        self.cuda_prg(*dev_data, grid=grid, block=block)        


def broadcast_binary_op(ary_1, ary_2, op: Expression):

    output_shape = np.broadcast_shapes(ary_1.shape, ary_2.shape)
    return ArrayExpression(
        expr=op((ary_1, ary_2)),
        shape=output_shape
    )

# }}}


# {{{

def exp(ary: LazyArray):
    expr = Call('exp', ary)
    return ArrayExpression(
        expr=expr,
        shape=ary.shape
    )


def log(ary: LazyArray):
    expr = Call('log', ary)
    return ArrayExpression(
        expr=expr,
        shape=ary.shape
    )

# }}}
