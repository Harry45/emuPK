# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Generate the set of training points to be used in the emulator
'''

import numpy as np

# our scripts
import utils.helpers as hp
import cosmoclass.spectrum as sp
import setemu as st
np.set_printoptions(suppress=True, precision=3)


def out_emu(module, parameters):
    '''
    Calculates the important quantities which we are going to emulate. These include
    - the growth function, A(z)
    - the linear mappter power spectrum P(k,z0)
    - q(k,z), a non-linear function

    P(k,z) = A(z) [1 + q(k,z)] P(k,z)

    :param: module (the CLASS class) - the CLASS module

    :param: parameters (np.ndarray) - set of input parameters

    :return: growth_function, linear, q
    '''

    linear, non_linear, gf = module.quantities(parameters)

    # calculate the non-linear function q
    dummy = non_linear / linear

    q = dummy / gf - 1.0

    # returns growth function as a flattened array
    gf = gf.flatten()

    # returns the linear power spectrum as a flatten array
    linear = linear.flatten()

    return gf, linear, q, non_linear


def main():

    # load input training points
    inputs = hp.load_arrays('processing/trainingpoints', 'scaled_inputs')
    npoints = inputs.shape[0]

    # get the cosmology module
    cosmo_module = sp.matterspectrum(zmax=4.66)
    cosmo_module.input_configurations()

    par = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 0.5692])
    g, l, q = out_emu(cosmo_module, par)

    print(q.flatten()[790:])

    # for i in range(npoints):

    #     # calculate important quantities
    #     # growth function
    #     # linear matter power spectrum at z0
    #     # q, the non linear function
    #     g, l, q = out_emu(cosmo_module, inputs[i])

    #     # store the quantities at each training point
    #     if st.gf_class:
    #         hp.store_arrays(g, 'processing/trainingpoints/g_class', 'g_' + str(i))
    #         hp.store_arrays(l, 'processing/trainingpoints/l_class', 'l_' + str(i))
    #         hp.store_arrays(q, 'processing/trainingpoints/q_class', 'q_' + str(i))

    #     else:
    #         hp.store_arrays(g, 'processing/trainingpoints/g', 'g_' + str(i))
    #         hp.store_arrays(l, 'processing/trainingpoints/l', 'l_' + str(i))
    #         hp.store_arrays(q, 'processing/trainingpoints/q', 'q_' + str(i))


if __name__ == "__main__":
    main()
