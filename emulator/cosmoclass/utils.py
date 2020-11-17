# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Some important functions for power spectrum calculation
'''

import numpy as np

def delete_module(module):
    '''
    Delete Class module - accumulates memory unnecessarily

    :param: module (classy.Class) - the Class module
    '''
    module.struct_cleanup()

    module.empty()

    del module


def dictionary_params(par: np.ndarray):
    '''
    A dictionary for storing all the parameters

    The parameters are organised in the following order:

    0: omega_cdm_h2
    1: omega_b_h2
    2: ln_10_10_A_s
    3: n_s
    4: h
    5: sum_neutrino
    6: A_bary

    :param: par (np.ndarray): parameters for the inference

    :return: cosmo (dict) : a dictionary for the cosmology setup

    :return: other (dict) : a dictionary for the neutrino settings

    :return: neutrino (dict) : a dictionary for the neutrino

    :return: nuisance (dict) : a dictionary which contains the baryon feedback parameter
    '''

    cosmo = {'omega_cdm': par[0], 'omega_b': par[1], 'ln10^{10}A_s': par[2], 'n_s': par[3], 'h': par[4]}

    other = {'N_ncdm': 1.0, 'deg_ncdm': 3.0, 'T_ncdm': 0.71611, 'N_ur': 0.00641}

    neutrino = {'m_ncdm': par[5] / other['deg_ncdm']}

    nuisance = {'A_bary': par[6]}

    return cosmo, other, neutrino, nuisance
