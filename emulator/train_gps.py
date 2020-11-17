# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Description : Routine to train all GPs in parallel for each specific type of power spectrum
'''

import os
import glob
import multiprocessing as mp
import numpy as np
import pandas as pd

# our scripts
import utils.helpers as hp
import optimisation as op


def training(inputs, file_name):
    '''
    Function to train GPs

    :param: inputs (np.ndarray) : array of size N_train x N_dim for the inputs to the GP

    :param: file_name (str) : name of the GP output, for example, pk_0
    '''

    print('Optimising GP for : {}'.format(file_name))

    outputs = hp.load_arrays('processing/trainingpoints/processed_pk', file_name)

    try:
        # train the GP
        gp_model = op.maximise(x_train=inputs, y_train=outputs)

        # store the GP in the specific folder
        hp.store_pkl_file(gp_model, 'gps', 'gp_' + file_name)

        # extract the marginal likelihood
        evidence = np.around(gp_model.min_chi_sqr, 0)

        # extract the kernel hyperparameters
        params = np.around(gp_model.opt_params, 3)

        with open('information/optimum.txt', 'a') as file:
            file.write('{0}\t{1}\t{2}\n'.format(file_name[3:], evidence, params))

    except BaseException:
        with open('information/debug.txt', 'a') as file:
            file.write('{0}\n'.format(file_name))


def worker(args):
    '''
    The argument here is simply the name of the output vector

    :param: args (list) : list containing the arguments to be fed for training GPs
    '''
    training(*args)


def final_sort(file_name='information/optimum.txt', save=True):
    '''
    When we train the GPs in parallel, they might be returned in the incorrect order. This function just sorts the file describing the marginal likelihood of the GPs in ascending order.

    :param: file_name (str) : name of the file (default: 'information/optimum.txt')

    :param: save (bool) if True, the sorted file will be saved under the name sorted_file.txt in the information folder

    :return: sorted_file (pd.DataFrame) : the sorted file
    '''
    sorted_file = pd.read_csv(file_name, sep='\t')

    sorted_file = sorted_file.sort_values(by='pk')

    if save:
        sorted_file.to_csv('information/sorted_file.txt', index=False, sep='\t')

    return sorted_file


def main():

    # the inputs are fixed
    inputs = hp.load_arrays('processing/trainingpoints', 'scaled_inputs')

    # number of gps
    ngps = len(os.listdir('processing/trainingpoints/processed_pk/'))

    try:

        # create the folder (to store information about GPs and training)
        if not os.path.exists('information'):
            os.makedirs('information')

        # if folder exists and there are previous files, remove them
        for filename in glob.glob("information/*.txt"):
            os.remove(filename)

        with open('information/optimum.txt', 'a') as file:
            file.write('{0}\t{1}\t{2}\n'.format('pk', 'evidence', 'params'))

    except BaseException:
        pass

    # make a list of arguments to pass to parallel processors
    arguments = [[inputs, 'pk_' + str(i)] for i in range(ngps)]

    # train GPs in parallel
    ncpu = mp.cpu_count()
    pool = mp.Pool(processes=ncpu)
    pool.map(worker, arguments)
    pool.close()

    # sort file file and overwrite it
    final_sort('information/optimum.txt', save=True)


# Standard boilerplate to call the main() function to begin the program.
if __name__ == '__main__':
    main()
