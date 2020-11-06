'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Functions to calculate the kernel matrix
'''

import numpy as np
from scipy.spatial.distance import cdist


def rbf(x_train, x_test=None, params=None):
    '''
    Implementation of the Radial Basis Function

    Inputs
    ------
    x_train (np.ndarray) : a matrix of size N x d (N > d)

    x_test (np.ndarray) : a matrix (or vector)

    params (np.ndarray) : kernel hyperparameters (amplitude and lengthscale)

    Returns
    -------
    kernel_matrix (np.ndarray) : the kernel matrix

    If the x_test is not part of the training set, following Rasmussen et al. (2006) the following will be returned:

    kernel_s (np.ndarray) : a vector of size N

    kernel_ss (np.ndarray) : a scalar (1 x 1) array
    '''

    # the amplitude and the lengthscales
    amp, scale = params[0], params[1:]

    if x_test is None:

        # calculate the pair-wise Euclidean distance
        distance = squared_distance(x_train, x_train, scale)

        # calculate the kernel matrix
        kernel_matrix = amp * np.exp(-0.5 * distance)

        return kernel_matrix

    else:

        # Ensure that x_test is a 2D array
        x_test = np.atleast_2d(x_test)

        # Compute pairwise distance between training point and test point
        distance1 = squared_distance(x_train, x_test, scale)

        # Compute distance with itself - this is just zero
        distance2 = np.zeros(1)

        # vector k_star
        kernel_s = amp * np.exp(-0.5 * distance1)

        # scaler k_star_star
        kernel_ss = amp * np.exp(-0.5 * distance2)

        return kernel_s, kernel_ss


def squared_distance(x1, x2, scale):
    '''
    Calculate the pairwise Euclidean distance between two input vectors (or matrix)

    Inputs
    ------
    x1 (np.ndarray) : first vector (or matrix if we have more than 1 training point)

    x2 (np.ndarray) : second vector (or matrix if we have more than 1 training point)

    scale (np.ndarray) : the characteristic lengthscales for the kernel

    Returns
    -------
    distance (np.ndarray) : pairwise Euclidean distance between the two vectors/matrix
    '''

    distance = cdist(x1 / scale, x2 / scale, metric='sqeuclidean')

    return distance
