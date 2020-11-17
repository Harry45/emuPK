# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Load and save the outputs (matter power spectrum) in the right format
'''

import os
import numpy as np
import pandas as pd

# out script
import utils.helpers as hp
import setemu as st


def process_output(outputs='processing/trainingpoints/power_spectrum', save=True):
    '''
    Process the output from CLASS so we can train each GP

    :param: outputs (str) : directory name where the outputs of CLASS are stored (default: 'processing/trainingpoints/power_spectrum')

    :param: save (bool) : if True, the processed outputs will be written to 'processing/trainingpoints/processed_pk/' and each output will be saved in the following format: pk_x (x is the i^th GP output)
    '''

    ntrain = len(os.listdir(outputs))

    record = []

    for i in range(ntrain):

        # load the outputs from CLASS
        # pk is arranged in (k,z)
        pk = hp.load_arrays(outputs, 'pk_' + str(i))

        # in record we append pk (flattened) which will be of size k x z, for example k = 40, z = 20
        record.append(pk.flatten())

    record = np.asarray(record)

    if save:

        n_gps = record.shape[1]

        for k in range(n_gps):
            hp.store_arrays(record[:, k], 'processing/trainingpoints/processed_pk', 'pk_' + str(k))

    return record


if __name__ == "__main__":

    power_spectra = process_output()
