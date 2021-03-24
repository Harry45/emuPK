'''
Module for important calculations involving the prior. For example,

- when scaling the Latin Hypercube samples to the appropriate prior range

- when calculating the posterior if the emulator is connected with an MCMC sampler
'''

import scipy.stats
import numpy as np
import settings as st


def entity(dictionary):
    '''
    Generates the entity of each parameter by using scipy.stats function.

    :param: dictionary (dict) - a dictionary containing information for each parameter, that is,

            - distribution, specified by the key 'distribution'

            - specifications, specified by the key 'specs'

    :return: dist (dict) - the distribution generated using scipy
    '''

    dist = eval('scipy.stats.' + dictionary['distribution'])(*dictionary['specs'])

    return dist


def all_entities(dict_params):
    '''
    Generate all the priors once we have specified them.

    :param: dict_params (dict) - a list containing the description for each parameter
    and each description (dictionary) contains the following information:

            - distribution, specified by the key 'distribution'

            - parameter name, specified by the key 'parameter'

            - specifications, specified by the key 'specs'

    :return: record (list) - a list containing the prior for each parameter, that is,
    each element contains the following information:

            - parameter name, specified by the key 'parameter'

            - distribution, specified by the key 'distribution'
    '''

    # create an empty list to store the distributions
    record = {}

    for c in dict_params:
        record[c] = entity(st.priors[c])

    return record


def log_prod_pdf(desc: dict, parameters: dict) -> float:
    '''
    Calculate the log-product for a set of parameters given the priors

    :param: desc (dict) - dictionary of parameters

    :param: parameters (np.ndarray) - an array of parameters

    :return:  log_sum (float) - the log-product of when the pdf of each parameter is multiplied with another
    '''

    # initialise log_sum to 0.0
    log_sum = 0.0

    # calculate the log-pdf for each parameter
    for p in parameters:
        log_sum += desc[p].logpdf(parameters[p])

    # if (any) parameter lies outside prior range, set log_sum to a very small value
    if np.isinf(log_sum):
        log_sum = -1E32

    return log_sum
