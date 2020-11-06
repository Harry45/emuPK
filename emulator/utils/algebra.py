'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Functions for linear algebra calculations
'''

import numpy as np
import scipy.linalg as sl
from GPy.util import linalg as gpl


def solve(matrix, b_vec, return_chol=False):
    '''
    Given a matrix and a vector, this solves for x in the following:

    Ax = b

    If A is diagonal, the calculations are simpler (do not require any inversions)

    Inputs
    ------
    matrix (np.ndarray) : 'A' matrix of size N x N

    b_vec (np.ndarray) : 'b' vector of size N

    return_chol (bool) : if True, the Cholesky factor will be retuned

    Returns
    -------
    dummy (np.ndarray) : 'x' in the equation above

    If return_chol is True,

    chol_factor (np.ndarray) : the Cholesky factor is returned
    '''

    if diagonal(matrix):

        # simple solution for x - no inversion
        dummy = 1. / np.atleast_2d(np.diag(matrix)).T * b_vec

        # if we want the Cholesky factor, it is a simple square root operation
        if return_chol:
            chol_factor = np.sqrt(matrix)
            return dummy, chol_factor
        else:
            return dummy

    else:

        # for stability, we use jitchol from the GPy package
        chol_factor = gpl.jitchol(matrix)

        # find x vector
        dummy = gpl.dpotrs(chol_factor, b_vec, lower=True)[0]

        if return_chol:
            return dummy, chol_factor
        else:
            return dummy


def matrix_inverse(matrix, return_chol=False):
    '''
    Sometimes, we would need the matrix inverse as well

    If we are dealing with diagonal matrix, inversion is simple

    Inputs
    ------
    matrix (np.ndarray) : matrix of size N x N

    return_chol (bool) : if True, the Cholesky factor will be returned

    Returns
    -------
    dummy (np.ndarray) : matrix inverse

    If return_chol is True,

    chol_factor (np.ndarray) : the Cholesky factor
    '''

    # check if matrix is diagonal first
    if diagonal(matrix):

        # simple matrix inversion
        dummy = np.diag(1. / np.diag(matrix))

        return dummy

    else:

        # calculate the Cholesky factor using jitchol from GPy
        # for numerical stability
        chol_factor = gpl.jitchol(matrix)

        # perform matrix inversion
        dummy = gpl.dpotrs(chol_factor, np.eye(chol_factor.shape[0]), lower=True)[0]

        if return_chol:
            return dummy, chol_factor
        else:
            return dummy


def diagonal(matrix):
    '''
    Check if a matrix is diagonal

    Inputs
    ------
    matrix (np.ndarray) : matrix of size N x N

    Returns
    -------
    cond (bool) : if diagonal, True
    '''

    if np.count_nonzero(matrix - np.diag(np.diagonal(matrix))) == 0:
        cond = True
        return cond
