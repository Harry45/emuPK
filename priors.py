'''
Author: (Dr to be) Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Priors for our parameters (either emulator or likelihood)
'''
import scipy.stats
import numpy as np


def entity(dictionary):
    '''
    Generates the entity of each parameter by using scipy.stats function.

    :param: dictionary (dict) - a dictionary containing information for each parameter, that is,
            - distribution, specificied by the key 'distribution'
            - parameter name, specified by the key 'parameter'
            - specifications, specified by the key 'specs'

    :return: param_entity (dict) - a dictionary containing the parameter name and the distribution generated
    '''

    dist = eval('scipy.stats.' + dictionary['distribution'])(*dictionary['specs'])

    param_entity = {'parameter': dictionary['parameter'], 'distribution': dist}

    return param_entity


def all_entities(list_params):
    '''
    Generate all the priors once we have specified them.

    :param: list_params (list) - a list containing the description for each parameter and each description (dictionary) contains the following information:

            - distribution, specificied by the key 'distribution'

            - parameter name, specified by the key 'parameter'

            - specifications, specified by the key 'specs'

    :return: record (list) - a list containing the prior for each parameter, that is, each element contains the following information:

            - parameter name, specifiied by the key 'parameter'
            
            - distribution, specified by the key 'distribution'
    '''

    # number of parameters
    n_params = len(list_params)

    # create an empty list to store the distributions
    record = []

    for i in range(n_params):
        record.append(entity(list_params[i]))

    return record


def log_prod_pdf(params_desc, parameters):
    '''
    Calculate the log-product for a set of parameters given the priors
    
    :param: params_desc (list) - list containing dictionaries of parameters. Each dictionary contains the parameter's name and its distribution.

    :param: parameters (np.ndarray) - an array of parameters

    :return:  log_sum (float) - the log-product of when the pdf of each parameter is multiplied with another
    '''

    # number of parameters
    n_params = len(parameters)

    # number of parameters should be the same as the length of the description for the parameters
    assert (len(params_desc) == n_params), 'Number of parameters should be of the same length as the prior list'

    # initialise log_sum to 0.0
    log_sum = 0.0

    # calculate the log-pdf for each parameter
    for i in range(n_params):
        log_sum += params_desc[i]['distribution'].logpdf(parameters[i])

    # if (any) parameter lies outside prior range, set log_sum to a very small value
    if np.isinf(log_sum):
        log_sum = -1E32

    return log_sum
