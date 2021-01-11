# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Description : Routine to train all GPs in parallel for each specific type of power spectrum
'''

import os
import multiprocessing as mp

# our scripts
import utils.helpers as hp
import optimisation as op
import setemu as st


def training(inputs, folder_name, file_name):
    '''
    Function to train GPs

    :param: inputs (np.ndarray) : array of size N_train x N_dim for the inputs to the GP

    :param: folder_name (str) : name of the folder where the outputs are stored, for example, out_g

    :param: file_name (str) : name of the GP output, for example, g_0
    '''

    print('Optimising GP for : {}'.format(file_name))

    outputs = hp.load_arrays('processing/trainingpoints/' + folder_name, file_name)

    try:
        # train the GP
        gp_model = op.maximise(x_train=inputs, y_train=outputs)

        # store the GP in the specific folder
        hp.store_pkl_file(gp_model, 'gps/' + folder_name, 'gp_' + file_name)

    except BaseException:
        pass


def worker(args):
    '''
    The argument here is simply the name of the output vector

    :param: args (list) : list containing the arguments to be fed for training GPs
    '''
    training(*args)


def main():

    # the inputs are fixed
    inputs = hp.load_arrays('processing/trainingpoints', 'scaled_inputs')

    # number of gps
    if st.gf_class:
        n_g = len(os.listdir('processing/trainingpoints/out_class_g/'))
        n_l = len(os.listdir('processing/trainingpoints/out_class_l/'))
        n_q = len(os.listdir('processing/trainingpoints/out_class_q/'))

        args_g = [[inputs, 'out_class_g', 'g_' + str(i)] for i in range(n_g)]
        args_l = [[inputs, 'out_class_l', 'l_' + str(i)] for i in range(n_l)]
        args_q = [[inputs, 'out_class_q', 'q_' + str(i)] for i in range(n_q)]

    else:
        n_g = len(os.listdir('processing/trainingpoints/out_g/'))
        n_l = len(os.listdir('processing/trainingpoints/out_l/'))
        n_q = len(os.listdir('processing/trainingpoints/out_q/'))

        args_g = [[inputs, 'out_g', 'g_' + str(i)] for i in range(n_g)]
        args_l = [[inputs, 'out_l', 'l_' + str(i)] for i in range(n_l)]
        args_q = [[inputs, 'out_q', 'q_' + str(i)] for i in range(n_q)]

    # train GPs in parallel
    ncpu = mp.cpu_count()

    # for the growth function
    # pool = mp.Pool(processes=ncpu)
    # pool.map(worker, args_g)
    # pool.close()

    # for the linear matter power spectrum
    pool = mp.Pool(processes=ncpu)
    pool.map(worker, args_l)
    pool.close()

    # for the q non linear function
    pool = mp.Pool(processes=ncpu)
    pool.map(worker, args_q)
    pool.close()


# Standard boilerplate to call the main() function to begin the program.
if __name__ == '__main__':
    main()
