import cantera as ct
import numpy as np
from minipyro.symbolic import Variable


def arrhenius_expr(rxn: ct.Reaction, temp: Variable):
    # Nonlinear functions
    exp = Variable('exp')
    log = Variable('log')
    # Arrhenius parameters
    log_a = np.log(rxn.rate.pre_exponential_factor)
    b = rxn.rate.temperature_exponent
    act_temp = -1 * rxn.rate.activation_energy / ct.gas_constant
    # Construct the Arrhenius expression
    return exp(log_a + b * log(temp) + act_temp / temp)
