# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Functions to transform the inputs and outputs
'''

import numpy as np


class transformation:

    '''
    Module to perform all relevant transformation, for example, pre-whitening the inputs and
    logarithm (supports log10 transformation) for the outputs.
    '''

    def __init__(self, theta: np.ndarray, y: np.ndarray):
        '''
        :param: theta (np.ndarray) : matrix of size N x d

        :param: y (np.ndarray) : a vector of the output

        :param: N is the number of training points

        :param: d is the dimensionality of the problem
        '''
        # input
        self.theta = theta

        msg = 'The number of training points is smaller than the dimension of the problem. Reshape your array!'

        assert self.theta.shape[0] > self.theta.shape[1], msg

        # dimension of the problem
        self.d = self.theta.shape[1]

        # number of training points
        self.N = self.theta.shape[0]

        # y is a vector of size N
        self.y = y.reshape(self.N, 1)

    def x_transform(self) -> np.ndarray:
        '''
        Transform the inputs (pre-whitening step)

        :return: theta_trans (np.ndarray) : transformed input parameters
        '''

        # calculate the covariance of the inputs
        cov = np.cov(self.theta.T)

        # calculate the Singular Value Decomposition
        a, b, c = np.linalg.svd(cov)

        # see PICO paper for this step
        m_diag = np.diag(1.0 / np.sqrt(b))

        # the transformation matrix
        self.mu_matrix = np.dot(m_diag, c)

        # calculate the transformed input parameters
        theta_trans = np.dot(self.mu_matrix, self.theta.T).T

        # store the transformed inputs
        self.theta_trans = theta_trans

        return theta_trans

    def x_transform_test(self, xtest: np.ndarray) -> np.ndarray:
        '''
        Given a test point, we transform the test point in the appropriate basis

        :param: xtext (np.ndarray) : a vector of dimension d for the test point

        :return: x_trans (np.ndarray) : the transformed input parameters
        '''

        # reshape the input
        xtest = xtest.reshape(self.d,)

        # tranform the input using the transformation matrix
        x_trans = np.dot(self.mu_matrix, xtest).reshape(1, self.d)

        return x_trans

    def y_transform(self) -> np.ndarray:
        '''
        Transform the output (depends on whether we want this criterion)

        If all the outputs are positive, then y_min = 0,
        otherwise the minimum is computed and the outputs are shifted by
        this amount before the logarithm transformation is applied

        :return: y_trans (np.ndarray) : array for the transformed output
        '''

        if (self.y > 0).all():

            # set the minimum to 0.0
            self.y_min = 0.0

            # calculate te logarithm of the outputs
            y_trans = np.log10(self.y)

            # store the transformed output
            self.y_trans = y_trans

            return y_trans

        else:
            # compute minimum y
            self.y_min = np.amin(self.y)

            # calcualte the logarithm of the outputs
            y_trans = np.log10(self.y - 2 * self.y_min)

            # store the transformed output
            self.y_trans = y_trans

            return y_trans

    def y_transform_test(self, y_original: np.ndarray) -> np.ndarray:
        '''
        Given a response/output which is not in the training set, this
        function will do the forward log_10 transformation.

        :param: y_original (float or np.ndarray) : original output

        :return: y_trans_test (array) : transformed output
        '''

        y_trans_test = np.log10(y_original - 2 * self.y_min)

        return y_trans_test

    def y_inv_transform_test(self, y_test: np.ndarray) -> np.ndarray:
        '''
        Given a response (a prediction), this function will do
        the inverse transformation (from log_10 to the original function).

        :param: y_test (float or np.ndarray) : a test (transformed) response (output)

        :return: y_inv (np.ndarray) : original (predicted) output
        '''

        y_inv = np.power(10, y_test) + 2.0 * self.y_min

        return y_inv
