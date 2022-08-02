# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Routine to train all GPs in parallel to emulate the 3D Power Spectrum
'''

import numpy as np
import multiprocessing as mp

# our scripts
import utils.common as uc
import utils.helpers as hp
import ml.optimisation as op
import settings as st

np.set_printoptions(precision=4, suppress=True)

logger = uc.get_logger('training', 'training', 'logs')


def train(cosmologies: np.ndarray, target: np.ndarray, folder_name: str, fname: str, kwargs: dict) -> None:
    '''
    Function to train GPs

    :param: cosmologies (np.ndarray) : array of size N_train x N_dim for the inputs to the GP

    :param: target (np.ndarray) : an array for the targets (function)

    :param: folder_name (str) : name of the folder where the outputs are stored

    :param: fname (str) : name of the GP output

    :param: kwargs (dict) : a dictionary with the settings for the GPs, for example, lambda_cap = 1000
    '''

    # Sometimes the GP might break due to numerical instrability
    # hence, try except

    try:
        # train the GP
        gp_model = op.maximise(cosmologies, target, **kwargs)

        # store the GP in the specific folder
        hp.store_pkl_file(gp_model, folder_name, fname)

        # add some important information to the log file
        info1 = folder_name + '/' + fname
        evidence = np.around(gp_model.min_chi_sqr, 4)

        logger.info(info1)
        logger.info(evidence)

    except BaseException:
        pass


def worker(args: list) -> None:
    '''
    The argument here is simply the name of the output vector

    :param: args (list) : list containing the arguments to be fed for training GPs
    '''
    train(*args)


def parallel_training(arguments: list) -> None:
    '''
    Call the parallel processing routine here

    :param: arguments (list) - list of arguments (inputs) to train the GPs

    :return: None
    '''

    # count number of CPU
    ncpu = mp.cpu_count()

    # run procedure in parallel
    pool = mp.Pool(processes=ncpu)
    pool.map(worker, arguments)
    pool.close()


def main(directory: str = 'semigps') -> None:
    '''
    Main function to train all the Gaussian Process models.

    :param: directory (str) - directory where the GPs are stored.

    :return: None
    '''

    # if we are using the 3 components and excluding neutrino
    if st.components and not st.neutrino:

        # the inputs are fixed
        cosmologies = hp.load_arrays('trainingset/components', 'cosmologies')

        # load the growth factor
        growth_factor = hp.load_arrays('trainingset/components', 'growth_factor')

        # load the q functions
        q_function = hp.load_arrays('trainingset/components', 'q_function')

        # load the linear matter power spectrum
        pk_linear = hp.load_arrays('trainingset/components', 'pk_linear')

        # number of GPs for each component
        n_gf = growth_factor.shape[1]
        n_qf = q_function.shape[1]
        n_pl = pk_linear.shape[1]

        # folders where we want to store the GPs
        folder_gf = directory + '/pknl_components' + st.d_one_plus + '/gf'
        folder_qf = directory + '/pknl_components' + st.d_one_plus + '/qf'
        folder_pl = directory + '/pknl_components' + st.d_one_plus + '/pl'

        # generate arguments to be fed to parallel routine
        # arguments for the growth factor
        arg_gf = [[cosmologies, growth_factor[:, i], folder_gf, 'gp_' + str(i), st.gf_args] for i in range(n_gf)]

        # arguments for the q(k,z) function
        if st.emu_one_plus_q:
            arg_qf = [[cosmologies, 1.0 + q_function[:, i], folder_qf, 'gp_' + str(i), st.qf_args] for i in range(n_qf)]
        else:
            arg_qf = [[cosmologies, q_function[:, i], folder_qf, 'gp_' + str(i), st.qf_args] for i in range(n_qf)]

        # arguments for linear matter power spectrum
        arg_pl = [[cosmologies, pk_linear[:, i], folder_pl, 'gp_' + str(i), st.pl_args] for i in range(n_pl)]

        # idea: should we emulate 1 + q(k,z) rather than q(k,z) to prevent Pk to be negative?
        # idea: should we use log-transformation for A(z) and q(k,z) or 1 + q(k,z)?
        # solution: option for the user to try these possibilities in any case;
        # perform training in parallel
        parallel_training(arg_gf)
        parallel_training(arg_qf)
        parallel_training(arg_pl)

    # if we are using the 3 components and excluding neutrino
    if st.components and st.neutrino:

        # the inputs are fixed
        cosmologies = hp.load_arrays('trainingset/components', 'cosmologies_neutrino')

        # load the growth factor
        growth_factor = hp.load_arrays('trainingset/components_neutrino', 'growth_factor')

        # load the q functions
        q_function = hp.load_arrays('trainingset/components_neutrino', 'q_function')

        # load the linear matter power spectrum
        pk_linear = hp.load_arrays('trainingset/components_neutrino', 'pk_linear')

        # number of GPs for each component
        n_gf = growth_factor.shape[1]
        n_qf = q_function.shape[1]
        n_pl = pk_linear.shape[1]

        # folders where we want to store the GPs
        folder_gf = directory + '/pknl_neutrino_components' + st.d_one_plus + '/gf'
        folder_qf = directory + '/pknl_neutrino_components' + st.d_one_plus + '/qf'
        folder_pl = directory + '/pknl_neutrino_components' + st.d_one_plus + '/pl'

        # generate arguments to be fed to parallel routine
        # arguments for the growth factor
        arg_gf = [[cosmologies, growth_factor[:, i], folder_gf, 'gp_' + str(i), st.gf_args] for i in range(n_gf)]

        # arguments for the q(k,z) function
        if st.emu_one_plus_q:
            arg_qf = [[cosmologies, 1.0 + q_function[:, i], folder_qf, 'gp_' + str(i), st.qf_args] for i in range(n_qf)]
        else:
            arg_qf = [[cosmologies, q_function[:, i], folder_qf, 'gp_' + str(i), st.qf_args] for i in range(n_qf)]

        # arguments for linear matter power spectrum
        arg_pl = [[cosmologies, pk_linear[:, i], folder_pl, 'gp_' + str(i), st.pl_args] for i in range(n_pl)]

        # perform training in parallel
        parallel_training(arg_gf)
        parallel_training(arg_qf)
        parallel_training(arg_pl)

    if not st.components and not st.neutrino:

        # the inputs are fixed
        cosmologies = hp.load_arrays('trainingset/pk_nonlinear', 'cosmologies')

        # the non linear matter power spectrum
        pk_nl = hp.load_arrays('trainingset/pk_nonlinear', 'pk_nl')

        # number of GPs
        npk = pk_nl.shape[1]

        # folder where we want to store the GPs
        folder_pknl = directory + '/pknl' + st.d_one_plus + '/'

        args = [[cosmologies, pk_nl[:, i], folder_pknl, 'gp_' + str(i), st.pknl_args] for i in range(npk)]

        # perform training in parallel
        parallel_training(args)

    if not st.components and st.neutrino:

        # the inputs are fixed
        cosmologies = hp.load_arrays('trainingset/pk_nonlinear', 'cosmologies_netrino')

        # the non linear matter power spectrum
        pk_nl = hp.load_arrays('trainingset/pk_nonlinear', 'pk_nl_neutrino')

        # number of GPs
        npk = pk_nl.shape[1]

        # folder where we want to store the GPs
        folder_pknl = directory + '/pknl_neutrino' + st.d_one_plus + '/'

        args = [[cosmologies, pk_nl[:, i], folder_pknl, 'gp_' + str(i), st.pknl_args] for i in range(npk)]

        # perform training in parallel
        parallel_training(args)


# Standard boilerplate to call the main() function to begin the program.
if __name__ == '__main__':
    main()
