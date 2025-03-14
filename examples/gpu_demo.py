import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from minipyro.codegen.python import get_thermochem_class
from minipyro.pyro_np import lazy_np


def run_minipyro():

    pyro_class = get_thermochem_class()
    pyro_gas = pyro_class(lazy_np)

    wg_size = 32
    num_x = 32 * wg_size
    temp = lazy_np.Placeholder(name='temperature', shape=(num_x, num_x))
    conc = lazy_np.Placeholder(name='concentration', shape=(2, num_x, num_x))

    rxn_rate = pyro_gas.get_rxn_rate(temp, conc)
    rxn_rate.compile('get_rxn_rate', wg_size)

    temp = 300 * np.ones((num_x, num_x))
    conc = 0.5 * np.ones((2, num_x, num_x))
    rate = np.zeros((num_x, num_x))
    rxn_rate.evaluate(conc, rate, temp)

    print(rxn_rate.cuda_code)
    return


if __name__ == '__main__':
    num_dev = drv.Device.count()
    if not num_dev:
        print('Found no devices, exiting gracefully.')
        exit()
    else:
        print(f'Found {num_dev} devices')
        
    print(f'Device name: {drv.Device(0).name()}')

    run_minipyro()
    exit()
