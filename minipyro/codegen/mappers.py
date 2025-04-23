from minipyro.symbolic import Variable


_prec = {'var': 0, 'call': 1, 'sum': 2, 'mul': 3, 'div': 4, 'sub': 5, 'pow': 6}


def parenthesize(expr_str, prec_expr, prec):
    if prec and prec > prec_expr:
        return f'({expr_str})'  # parenthesize result
    else:
        return expr_str


# {{{ Code generation

class CodeGenerationMapper:

    def rec(self, expr, *args):
        import numbers
        if isinstance(expr, numbers.Number):
            return f'{expr}'
        if args:
            return getattr(self, expr.mapper_method)(expr, *args)
        else:
            return getattr(self, expr.mapper_method)(expr, None)

    def map_variable(self, expr, prec):
        return expr.name

    def map_sum(self, expr, prec):
        return parenthesize(
            ' + '.join([self.rec(c, _prec['sum']) for c in expr.children]),
            _prec['sum'], prec)

    def map_product(self, expr, prec):
        return parenthesize(
            ' * '.join([self.rec(c, _prec['mul']) for c in expr.children]),
            _prec['mul'], prec)

    def map_power(self, expr, prec):
        return parenthesize(
            ' ** '.join([
                self.rec(expr.base), self.rec(expr.exponent)
            ]),
            _prec['pow'], prec
        )
    
    def map_quotient(self, expr, prec):
        return parenthesize(
            ' / '.join([
                self.rec(expr.num), self.rec(expr.den)
            ]),
            _prec['div'], prec
        )

    def map_subscript(self, expr, prec):
        ids = expr.i.name if isinstance(expr.i, Variable) else str(expr.i)
        return '{:s}[{:s}]'.format(self.rec(expr.a, _prec['sub']), ids)

    def map_call(self, expr, prec):
        return 'self.pyro_np.{:s}({:s})'.format(
            self.rec(expr.fn_name, _prec['call']),
            self.rec(expr.fn_arg, _prec['call'])
        )

# }}}


# {{{

class LoopyMapper:
    prec = {'var': 0, 'call': 1, 'sum': 2,
            'mul': 3, 'div': 4, 'sub': 5, 'pow': 6}

    def rec(self, ary, *args):
        import numbers
        if isinstance(ary, numbers.Number):
            return f'{ary}'
        if args:
            return getattr(self, ary.expr.mapper_method)(ary, *args)
        else:
            return getattr(self, ary.expr.mapper_method)(ary, None)

    def map_sum(self, ary, prec):
        return parenthesize(
            ' + '.join([
                self.rec(c, prec) for c in ary.expr.children
            ]), _prec['sum'], prec)

    def map_product(self, ary, prec):
        return parenthesize(
            ' * '.join([
                self.rec(c, prec) for c in ary.expr.children
            ]), _prec['mul'], prec)

    def map_power(self, ary, prec):
        return parenthesize(
            ' ** '.join([
                self.rec(ary.expr.base, prec),
                self.rec(ary.expr.exponent, prec)
            ]), _prec['pow'], prec)
    
    def map_quotient(self, ary, prec):
        return parenthesize(
            ' / '.join([
                self.rec(ary.expr.num, prec), self.rec(ary.expr.den, prec)
            ]), _prec['div'], prec)

    def map_subscript(self, ary, prec):
        dim = len(ary.shape)
        idx = (str(ary.expr.i),) + tuple(f'i{i}' for i in range(dim))
        return ary.expr.a.name + '[{:s}]'.format(', '.join([i for i in idx]))

    def map_variable(self, ary, prec):
        dim = len(ary.shape)
        idx = tuple(f'i{i}' for i in range(dim))
        return ary.name + '[{:s}]'.format(', '.join([i for i in idx]))

    def map_call(self, ary, prec):
        return ary.expr.fn_name + '({:s})'.format(
            self.rec(ary.expr.fn_arg, _prec['call'])
        )

# }}}
