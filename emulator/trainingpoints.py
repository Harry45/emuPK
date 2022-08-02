# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

"""
Routine to scale the Latin Hypercube samples according to the prior and evaluate the power spctrum at these points.
"""

from typing import Tuple
import timeit
import numpy as np
import pandas as pd

# our Python scripts
import priors as pr
import utils.helpers as hp
import utils.common as uc
import settings as st
import cosmology.spectrumclass as sp
import cosmology.cosmofuncs as cf

logger = uc.get_logger('trainingpoints', 'class_runs_neutrino', 'logs')


def CLASS_RUN(module: object, parameter: np.ndarray, index: int) -> Tuple[bool, dict]:
    '''
    Run CLASS given an input parameter to generate the training points (outputs)

    :param: module (object) - the CLASS module

    :param: parameter (np.ndarray) - the input cosmology, either 5 dimensions or 6 dimensions

    :index: i*th cosmology from the LHS file

    :return: state (bool), quantities (dict) - state indicates if the run is successful, quantities contain the important quantities generated
    '''

    # input to CLASS is a dictionary
    param = cf.mk_dict(st.cosmology, parameter)

    # for recording the parameter information (to 4 decimal places)
    p_info = cf.mk_dict(st.cosmology, np.around(parameter, 4))

    info1 = 'Cosmology {0:4d} : {1}'.format(index, p_info)
    logger.info(info1)

    # generate all the quantities
    # previously: quantities = cosmo_module.pk_nonlinear_components(param)
    start_time = timeit.default_timer()
    state, quantities = cf.runTime(cf.timeOutComponents, module, param, st.timeout)
    elapsed = timeit.default_timer() - start_time

    info2 = 'Cosmology {0:4d} : Time taken is {1:.2f} seconds'.format(index, elapsed)
    logger.info(info2)

    return state, quantities


class trainingset(object):
    '''
    Runs CLASS at the LHS points generated using the maximin procedure. If we want to sample the neutrino mass, then, please use maximin_1000_6D as input (assuming it has already been generated), otherwise please use maximin_1000_5D.

    '''

    def __init__(self, lhs: str = 'maximin_1000_6D'):
        '''
        Generates the training set (inputs and outputs) but we need to scale the LHS samples
        first before evaluating the matter power spectrum using CLASS

        :param: lhs (str) - the reference to the file containing the generated LHS points
        '''

        self.lhs = lhs

        # load the LHS points
        self.inputs = pd.read_csv('lhs/' + self.lhs, index_col=0).values

        # number of cosmologies
        self.ncosmo = self.inputs.shape[0]

        # number of dimensions
        self.ndim = self.inputs.shape[1]

        # dimensions should agree according to the setting file
        assert self.ndim == len(st.cosmology), 'Dimension mis-match!'

    def scale(self, save: bool = True) -> np.ndarray:
        '''
        Scale the LHS according to the prior range. See setting file to set up the priors for the LHS samples.

        :param: save (bool) - if True, the scaled inputs (cosmologies) will be written to a file

        :return: cosmologies (np.ndarray) - the scaled inputs
        '''

        # generate the list of distributions
        all_priors = pr.all_entities(st.priors)

        # create an empty array to store the scaled parameters
        cosmologies = np.zeros_like(self.inputs)

        for i, p in enumerate(st.cosmology):
            cosmologies[:, i] = all_priors[p].ppf(self.inputs[:, i])

        if save:
            if st.neutrino:
                hp.store_arrays(cosmologies, 'trainingset', 'cosmologies_neutrino')
            else:
                hp.store_arrays(cosmologies, 'trainingset', 'cosmologies')

        return cosmologies

    def targets(self, cosmologies: np.ndarray, save: bool = False) -> np.ndarray:
        '''
        Generate the power spectrum at the specfic cosmologies

        :param: save (bool) - if True, the generated power spectrum will be saved in a directory. Note that the power spectrum is of shape (nk x nz), for example, 40 x 20. So the final shape will be of size (ncosmo x nk x nz). The power spectrum is flattened in this case, so we save a file of size 1000 x 800 (ncosmo = 1000, nk = 40, nz = 20). Therefore, we will have 800 separate GPs in this example.

        :param: cosmologies (np.ndarray) - set of cosmologies where we want to run CLASS

        :param: save (bool) - if True, the generated targets (training points/ power spectrum) will be saved in a directory

        :return: components (dict) - a list of the different quantities (growth factor, linear matter power spectrum, q function) evaluated at different cosmologies or

        :return: pk_non (np.ndarray) - the power spectrum evaluated at each cosmology
        '''

        # get the class module
        cosmo_module = sp.powerclass()
        cosmo_module.configurations()

        if st.components:

            # empty list for different quantities
            # the growth factor
            growth_factor = []

            # the linear matter power spectrum
            pk_linear = []

            # the q(k,z) function
            q_function = []

            # set of cosmologies successfully calculated
            class_cosmo = []

            for i in range(self.ncosmo):

                state, quantities = CLASS_RUN(cosmo_module, cosmologies[i], i)

                if quantities is None:

                    info3 = 'CLASS cannot compute Pk at this point - adding small perturbation to inputs'
                    logger.warning(info3)

                    ntrial = 0

                    while state is False:

                        # update cosmology by a jitter term
                        cosmologies[i] += 1E-4 * np.random.randn(self.ndim)

                        state, quantities = CLASS_RUN(cosmo_module, cosmologies[i], i)

                        ntrial += 1

                    logger.info('Number of attempts is {}'.format(ntrial))

                # record all quantities
                growth_factor.append(quantities['gf'].flatten())
                pk_linear.append(quantities['pl'].flatten())
                q_function.append(quantities['qf'].flatten())
                class_cosmo.append(cosmologies[i])

            if save:
                if st.neutrino:
                    hp.store_arrays(growth_factor, 'trainingset/components', 'growth_factor_neutrino')
                    hp.store_arrays(pk_linear, 'trainingset/components', 'pk_linear_neutrino')
                    hp.store_arrays(q_function, 'trainingset/components', 'q_function_neutrino')
                    hp.store_arrays(class_cosmo, 'trainingset/components', 'cosmologies_neutrino')

                else:
                    hp.store_arrays(growth_factor, 'trainingset/components', 'growth_factor')
                    hp.store_arrays(pk_linear, 'trainingset/components', 'pk_linear')
                    hp.store_arrays(q_function, 'trainingset/components', 'q_function')
                    hp.store_arrays(class_cosmo, 'trainingset/components', 'cosmologies')

            components = {'growth_factor': growth_factor, 'pk_linear': pk_linear, 'q_function': q_function}

            return components

        else:

            # create an empty list to record the power spectrum
            pk_non = []

            # generate the targets
            for i in range(self.ncosmo):

                # input to CLASS is a dictionary
                param = cf.mk_dict(st.cosmology, cosmologies[i])

                pk = cosmo_module.pk_nonlinear(param).flatten()
                pk_non.append(pk)

            # save the power spectrum
            if save:
                if st.neutrino:
                    hp.store_arrays(pk_non, 'trainingset/pk_nonlinear', 'pk_nl_neutrino')
                else:
                    hp.store_arrays(pk_non, 'trainingset/pk_nonlinear', 'pk_nl')

            return pk_non


if __name__ == "__main__":

    training_points = trainingset(lhs='maximin_1000_5D')
    cosmologies = training_points.scale(save=False)
    outputs = training_points.targets(cosmologies, save=True)
