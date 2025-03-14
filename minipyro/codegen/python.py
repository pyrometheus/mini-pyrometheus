import cantera as ct
from mako.template import Template
from minipyro.chem_expr import arrhenius_expr
from minipyro.symbolic import Variable
from minipyro.codegen.mappers import CodeGenerationMapper


code_tpl = Template("""
import numpy as np


class Thermochemistry:

    def __init__(self, pyro_np=np):
        self.pyro_np = pyro_np

    def get_rxn_rate(self, temperature, concentration):
        k = ${cgm.rec(arrhenius_expr(rxn, Variable("temperature")))}
        conc_product = ${cgm.rec(conc_product)}
        return k * conc_product
""", strict_undefined=True)


def get_thermochem_class():

    rxn = ct.Reaction(
        # Reactants & Products for M + N -> P + Q
        {'m': 1, 'n': 1}, {'p': 1, 'q': 1},
        # Arrhenius coefficients, taken from Reaction 1, San Diego mech
        {'A': 35127309770106.477, 'b': -0.7, 'Ea': 8590 * ct.gas_constant}
    )

    conc = Variable('concentration')
    conc_product = conc[0] * conc[1]

    cgm = CodeGenerationMapper()
    code_str = code_tpl.render(
        Variable=Variable,
        arrhenius_expr=arrhenius_expr,
        conc_product=conc_product,
        rxn=rxn,
        cgm=cgm,
    )

    exec_dict = {}
    exec(compile(code_str, '<generated code', 'exec'), exec_dict)
    exec_dict['_MODULE_SOURCE_CODE'] = code_str
    return exec_dict['Thermochemistry']
