# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Perform all additional operations such as prediction, interpolation, gradient calculation for GPs
'''

import numpy as np
import scipy.interpolate as itp


def kz_interpolate(inputs: list, grid: list) -> np.ndarray:
    '''
    Function to perform 2D interpolation using interpolate.interp2d

    :param: inputs (list) : inputs to the interpolation module, that is, we need to specify the following:
        - x
        - y
        - f(x,y)
        - 'linear', 'cubic', 'quintic'

    :param: grid (list) : a list containing xnew and ynew

    :return: pred_new (np.ndarray) : the predicted values on the 2D grid
    '''

    # transform in inputs to log
    k, z, f_kz, int_type = np.log(inputs[0]), inputs[1], np.log(inputs[2]), inputs[3]

    inputs_trans = [k, z, f_kz, int_type]

    # tranform the grid to log
    knew, znew = np.log(grid[0]), grid[1]

    grid_trans = [knew, znew]

    f = itp.interp2d(*inputs_trans)

    pred_new = np.exp(f(*grid_trans))

    return pred_new


def ps_interpolate(inputs: list) -> np.ndarray:
    '''
    Function to interpolate the power spectrum along the redshift axis

    :param: inputs (list or tuple) : x values, y values and new values of x

    :return: ynew (np.ndarray) : an array of the interpolated power spectra
    '''

    x, y, xnew = np.log(inputs[0]), np.log(inputs[1]), np.log(inputs[2])

    spline = itp.splrep(x, y)

    ynew = np.exp(itp.splev(xnew, spline))

    return ynew


def prediction(input_pred: list) -> float:
    '''
    For each GP we have to calculate the mean prediction

    :param: input_pred (list or tuple): array for the test point and whole gp module

    :param: mean_pred (float) : the mean prediction from the GP
    '''

    testpoint, gp = input_pred[0], input_pred[1]

    mean_pred = gp.pred_original_function(testpoint).reshape(1,)

    return mean_pred[0]


def gradient(input_pred: list) -> float:
    '''
    For each GP we have to calculate the mean prediction

    :param: input_pred (list or tuple): array for the test point and whole gp module

    :param: mean_pred (float) : the mean prediction from the GP
    '''

    testpoint, gp, order = input_pred[0], input_pred[1], input_pred[2]

    if order == 1:
        first_der = gp.derivatives(testpoint, order)

        return first_der

    else:
        first_der, second_der = gp.derivatives(testpoint, order)

        return first_der, second_der
