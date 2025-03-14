import cantera as ct
from mako.template import Template
from minipyro.chem_expr import arrhenius_expr
from minipyro.codegen.mappers import CodeGenerationMapper
from minipyro.symbolic import Variable


# {{{ Codegen Mapper

class FortranMapper(CodeGenerationMapper):

    def map_subscript(self, expr, prec):
        ids = expr.i.name if isinstance(expr.i, Variable) else str(expr.i + 1)
        return '{:s}({:s})'.format(self.rec(expr.a, self.prec['sub']), ids)

    def map_call(self, expr, prec):
        return '{:s}({:s})'.format(
            self.rec(expr.fn_name, self.prec['call']),
            self.rec(expr.fn_arg, self.prec['call'])
        )

# }}}


# {{{ Code template

code_tpl = Template("""

module Thermochemistry

    implicit none

contains

    subroutine get_rxn_rate(temperature, concentration, rxn_rate)

        real(dp), intent(in) :: temperature
        real(dp), intent(in) :: concentration(2)
        real(dp), intent(out) :: rxn_rate

        real(dp) :: k, conc_product

        k = ${cgm.rec(arrhenius_expr(rxn, Variable("temperature")))}
        conc_product = ${cgm.rec(conc_product)}
        rxn_rate = k * conc_product

    end subroutine
""", strict_undefined=True)

# }}}


def get_thermochem_class(write_path):

    rxn = ct.Reaction(
        # Reactants & Products for M + N -> P + Q
        {'m': 1, 'n': 1}, {'p': 1, 'q': 1},
        # Arrhenius coefficients, taken from Reaction 1, San Diego mech
        {'A': 35127309770106.477, 'b': -0.7, 'Ea': 8590 * ct.gas_constant}
    )

    conc = Variable('concentration')
    conc_product = conc[0] * conc[1]

    cgm = FortranMapper()
    code_str = code_tpl.render(
        Variable=Variable,
        arrhenius_expr=arrhenius_expr,
        conc_product=conc_product,
        rxn=rxn,
        cgm=cgm,
    )
    with open(write_path + 'demo_codegen.f90', 'w') as fh:
        print(code_str, file=fh)
