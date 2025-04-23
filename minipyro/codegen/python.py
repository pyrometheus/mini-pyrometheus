from bandit.general_thermochem import BaseMechanism
from mako.template import Template
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE, PREC_CALL
# from minipyro.symbolic import Variable
# from minipyro.codegen.mappers import CodeGenerationMapper


class CodeGenerationMapper(StringifyMapper):
    def map_constant(self, expr, enclosing_prec):
        return repr(expr)

    def map_call(self, expr, enclosing_prec, *args, **kwargs):
        return self.format(
            "self.pyro_np.%s(%s)",
            self.rec(expr.function, PREC_CALL, *args, **kwargs),
            self.join_rec(", ", expr.parameters, PREC_NONE, *args, **kwargs),
        )


code_tpl = Template("""
import numpy as np


class Thermochemistry:

    def __init__(self, pyro_np=np):
        self.pyro_np = pyro_np

    def get_rxn_rate(self, temperature, concentrations):
        return ${cgm(mass_action_rate)}
""", strict_undefined=True)


def get_thermochem_class(bandit_mech: BaseMechanism):

    cgm = CodeGenerationMapper()
    code_str = code_tpl.render(
        mass_action_rate=bandit_mech.mass_action_rates[0],
        cgm=cgm,
    )

    exec_dict = {}
    exec(compile(code_str, '<generated code', 'exec'), exec_dict)
    exec_dict['_MODULE_SOURCE_CODE'] = code_str
    return exec_dict['Thermochemistry'], code_str
