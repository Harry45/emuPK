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
np.set_printoptions(suppress=True, precision=3)


def summary_statistics(train_points):
    '''
    Calculates the summaries of the training points generated from CLASS

    :param: train_points (np.ndarray) - an array of size n_train, the number of training points

    :return: df (pandas dataframe) - a dataframe with the following quantities:

     - min
     - max
     - mean
     - standard deviation
    '''

    # mean
    m = np.mean(train_points)

    # standard deviation
    s = np.std(train_points)

    # minimum
    minimum = np.min(train_points)

    # maximum
    maximum = np.max(train_points)

    # quantities
    quant = np.atleast_2d([m, s, minimum, maximum])

    df = pd.DataFrame(quant, columns=['mean', 'std', 'min', 'max'])

    return df


def process_output(outputs='processing/trainingpoints/', save=False):
    '''
    Process the output from CLASS so we can train each GP

    :param: outputs (str) : directory name where the outputs of CLASS are stored (default: 'processing/trainingpoints/power_spectrum')

    :param: save (bool) : if True, the processed outputs will be written to 'processing/trainingpoints/processed_pk/' and each output will be saved in the following format: pk_x (x is the i^th GP output)
    '''

    ntrain = len(os.listdir(outputs + 'l'))

    l_train = []
    g_train = []
    q_train = []

    for i in range(ntrain):

        # load the outputs from CLASS
        # l has shape nk
        # g is shape nz
        # q is arranged in (k,z)

        if st.gf_class:
            l = hp.load_arrays(outputs + 'l_class', 'l_' + str(i))
            g = hp.load_arrays(outputs + 'g_class', 'g_' + str(i))
            q = hp.load_arrays(outputs + 'q_class', 'q_' + str(i))

        else:
            l = hp.load_arrays(outputs + 'l', 'l_' + str(i))
            g = hp.load_arrays(outputs + 'g', 'g_' + str(i))
            q = hp.load_arrays(outputs + 'q', 'q_' + str(i))

        # in record we append pk (flattened) which will be of size k x z, for example k = 40, z = 20
        q_train.append(q.flatten())
        l_train.append(l)
        g_train.append(g)

    q_train = np.asarray(q_train)
    l_train = np.asarray(l_train)
    g_train = np.asarray(g_train)

    # number of outputs for each output from CLASS
    n_g = g_train.shape[1]
    n_l = l_train.shape[1]
    n_q = q_train.shape[1]

    summaries = []
    for i in range(q_train.shape[1]):
        summaries.append(summary_statistics(q_train[:, i]))

    summaries = pd.concat(summaries).reset_index(drop=True)

    if save:

        if st.gf_class:
            # save the non-linear function q
            for i in range(n_q):
                hp.store_arrays(q_train[:, i], outputs + 'out_class_q', 'q_' + str(i))

            # save the linear matter power spectrum
            for i in range(n_l):
                hp.store_arrays(l_train[:, i], outputs + 'out_class_l', 'l_' + str(i))

            # save the non-linear function q
            for i in range(n_g):
                hp.store_arrays(g_train[:, i], outputs + 'out_class_g', 'g_' + str(i))

            # save the summaries
            hp.save_excel(summaries, outputs + 'summaries', 'summaries_class')

        else:

            # save the non-linear function q
            for i in range(n_q):
                hp.store_arrays(q_train[:, i], outputs + 'out_q', 'q_' + str(i))

            # save the linear matter power spectrum
            for i in range(n_l):
                hp.store_arrays(l_train[:, i], outputs + 'out_l', 'l_' + str(i))

            # save the non-linear function q
            for i in range(n_g):
                hp.store_arrays(g_train[:, i], outputs + 'out_g', 'g_' + str(i))

            # save the summaries
            hp.save_excel(summaries, outputs + 'summaries', 'summaries')


if __name__ == "__main__":

    process_output(save=True)
