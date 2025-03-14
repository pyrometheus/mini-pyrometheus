import numpy as np
from minipyro.codegen.python import get_thermochem_class
from minipyro.pyro_np import adiff_np


def print_gradients(g):
    nm_sym = {'temperature': 'T', 'concentration': 'C'}
    for n in g:
        print(51*'_' + f' dR/d{nm_sym[n]} ' + 51 * '_' + '\n')
        if len(g[n].shape) == 1:
            print('[{:s}]'.format(
                ', '.join(['{:.3e}'.format(r) for r in g[n]])
            ))
        else:
            for row in g[n]:
                print('[{:s}]'.format(', '.join([
                    '{:.3e}'.format(r) for r in row
                ])))
            print('\n\n')


def run_minipyro():

    pyro_class = get_thermochem_class()
    pyro_gas = pyro_class(adiff_np)

    num_x = 10
    temp_np = 300 * np.ones((num_x,), dtype=np.float64)
    conc_np = 0.5 * np.ones((2, num_x,), dtype=np.float64)

    temp_ad = adiff_np.AutodiffVariable(temp_np, name='temperature')
    conc_ad = adiff_np.AutodiffVariable(conc_np, name='concentration')
    rxn_rate = pyro_gas.get_rxn_rate(temp_ad, conc_ad)
    print('R = [{:s}]'.format(
        ', '.join(['{:.3e}'.format(r) for r in rxn_rate.values])
    ))

    g = rxn_rate.gradient()

    print_gradients(g)    
    return


if __name__ == '__main__':
    run_minipyro()
    exit()
