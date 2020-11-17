# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Scale the training points according to the prior range
'''


import numpy as np
import pandas as pd


import utils.helpers as hp
import priors as pr
import setemu as st


def scale_points(lhs_points, save=True):
    '''
    Scale the training points according to the priors specified

        :param: lhs_points (str) - name of the file containing the LHS points

        :param: save (bool) - if True, the scaled inputs will be saved in a directory

        :return: scaled (np.ndarray) - the scaled inputs
    '''

    inputs = pd.read_csv('processing/design/' + lhs_points, index_col=0).values

    # generate the list of distributions
    all_priors = pr.all_entities(st.emu_params)

    # create an empty array to store the scaled parameters
    scaled = np.zeros_like(inputs)

    for i in range(len(all_priors)):
        scaled[:, i] = all_priors[i]['distribution'].ppf(inputs[:, i])

    if save:
        hp.store_arrays(scaled, 'processing/trainingpoints', 'scaled_inputs')

    return scaled


if __name__ == "__main__":

    X_prime = scale_points('maximin_1000_7D', save=True)
