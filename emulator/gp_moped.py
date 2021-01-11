'''
Authors: Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
Affiliation: Imperial College London
Department: Imperial Centre for Inference and Cosmology
Email: a.mootoovaloo17@imperial.ac.uk

Description:
Gaussian Process script for emulating the KiDS-450 MOPED coefficients

This code has been tested in Python 3 version 3.7.4
'''

import logging
import logging.handlers
import os
import numpy as np
import scipy.optimize as op
from scipy.spatial.distance import cdist
from GPy.util import linalg as gpl


# ignore some numerical errors
# and print floats at a certain precision
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, suppress=False)


# Settings for our log file
FORMAT = "%(levelname)s:%(filename)s.%(funcName)s():%(lineno)-8s %(message)s"
filename = "logs/gaussian_process_moped.log"


class GP:
    def __init__(self, data, sigma=[-4.0], train=False, nrestart=5, ndim=6):
        '''GP class

        :param data (np.ndarray): size N x ndim (dimension of the problem) + 1.
                The first ndim columns contain the inputs to the GP and the
                last column contains the output

        :param sigma (np.ndarray): size N or 1. We assume noise-free
                regression. Default: [-5.0] and this is log-standard
                deviation. This is also referred to as the jitter term
                in GP for numerical stability. At this point, the code
                does not support full noise covariance matrix.

        :param train (bool): True indicates that we will train the
                GP, otherwise, it uses the default values of the
                kernel parameters

        :param nrestart (int): Number of times we want to restart the optimiser

        :param ndim (int): the dimension of the problem
        '''

        if not os.path.exists('logs'):
            os.mkdir('logs')

        should_roll_over = os.path.isfile(filename)
        handler = logging.handlers.RotatingFileHandler(
            filename, mode='w', backupCount=0)

        # log already exists, roll over
        if should_roll_over:
            handler.doRollover()

        # Create our log file
        logging.basicConfig(
            filename=filename, level=logging.DEBUG, format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initialising variables...')

        # data
        data = np.array(data)

        # inputs
        self.theta = data[:, 0:-1]

        # outputs
        self.y_ = data[:, -1]

        # Transformation
        self.max_y = np.max(self.y_)
        self.min_y = np.min(self.y_)

        if self.min_y > 0:
            self.constant = 0.0
        else:
            self.constant = -2.0 * self.min_y

        print('The offset is {0:.2f}'.format(self.constant))

        # log_10 transformation
        self.y_ = np.log10(self.y_ + self.constant)

        # compute mean
        self.mean_y = np.mean(self.y_)

        # compute standard deviation
        self.std_y = np.std(self.y_)

        # standardize the log MOPED coefficients
        self.y = (self.y_ - self.mean_y) / self.std_y

        # Numer of training points
        self.N = len(self.y)

        # the noise term: the jitter term
        self.sigma = np.array(sigma)

        # training
        self.train = train

        # number of restart
        self.nrestart = nrestart

        # dimension of the problem
        self.d = self.theta.shape[1]
        self.ndim = ndim

        assert self.d == self.ndim, 'Mis-match in input dimension'

    def transform(self):
        '''
        Function to pre-whiten the input parameters

        Args:
            None: uses the inputs to the class method above

        Returns:
            None: we store the transformation matrix
            and the transformed parameters
        '''

        # first compute the covariance matrix of the inputs
        cov = np.cov(self.theta.T)

        # then compute the SVD of the covariance matrix
        a, b, c = np.linalg.svd(cov)

        # Compute 1/square-root of the eigenvalues
        M = np.diag(1.0 / np.sqrt(b))

        # compute and store the transformation matrix
        self.MU = np.dot(M, c)

        self.logger.info(
            'Size of the transformation matrix is {}'.format(self.MU.shape))

        # compute the transformed parameters
        self.theta_ = np.dot(self.MU, self.theta.T)

        # transpose - to get the N x d matrix back
        self.theta_ = self.theta_.T

    def rbf(self, label, X1, X2=None):
        '''
        Function to generate the RBF kernel

        Args:
            (str) label: 'trainSet', 'trainTest' and 'testSet'
                        same notations as in Rasmussen (2006)

            (array) X1: first N x d inputs

            (array) X2: second set of N x d array - can
                        either be training set or test point

        Returns:
            (array) : either K or Ks or Kss

        '''

        # Amplitude of kernel
        amp = np.exp(2.0 * self.width)

        # Divide inputs by respective characteristic lengthscales
        X1 = X1 / np.exp(self.scale)
        X2 = X2 / np.exp(self.scale)

        # Compute pairwise squared euclidean distance
        distance = cdist(X1, X2, metric='sqeuclidean')

        # Generate kernel or vector (if test point)
        if label == 'trainSet':
            K = amp * np.exp(-0.5 * distance)
            return K

        elif label == 'trainTest':
            Ks = amp * np.exp(-0.5 * distance)
            Ks = Ks.flatten()
            Ks = Ks.reshape(len(Ks), 1)
            return Ks

        elif label == 'testSet':
            Kss = amp
            Kss = Kss.reshape(1, 1)
            return Kss

    def distance_per_dim(self, x1, x2):
        '''
        Function to compute pairwise distance for each dimension

        Args:
            (array) x1: a vector of length equal to the number
                    of training points

            (array) x2: a vector of length equal to the number
                    of training points

        Returns:
            (array) a matrix of size N x N
        '''

        # reshape first and second vector in
        # the right format
        x1 = x1.reshape(len(x1), 1)
        x2 = x2.reshape(len(x2), 1)

        # compute pairwise squared euclidean distance
        D = cdist(x1, x2, metric='sqeuclidean')

        return D

    def kernel(self, label, X1, X2=None):
        '''
        Function to compute the kernel matrix

        Args:
            (str) label: 'trainSet', 'trainTest' and 'testSet'
                        same notations as in Rasmussen (2006)

            (array) X1: first N x d inputs

            (array) X2: second set of N x d array - can
                        either be training set or test point

        Returns:
            (array) : either K or Ks or Kss

        '''

        if label == 'trainSet':
            K = self.rbf('trainSet', X1, X2)
            np.fill_diagonal(K, K.diagonal() + np.exp(2.0 * self.sigma))
        else:
            K = self.rbf(label, X1, X2)
        return K

    def alpha(self):
        '''
        Function to compute alpha = K^-1 y

        Args:
            None

        Returns:
            (array) alpha of size N x 1
        '''

        # compute the kernel matrix of size N x N
        K = self.kernel('trainSet', self.theta_, self.theta_)

        # compute the Cholesky factor
        self.L = gpl.jitchol(K)

        # Use triangular method to solve for alpha
        alp = gpl.dpotrs(self.L, self.y, lower=True)[0]

        return alp

    def cost(self, theta):
        '''
        Function to calculate the negative log-marginal likelihood
        (cost) of the Gaussian Process

        Args:
            (array) theta: the kernel hyperparameters

        Returns:
            (array) cost: outputs the cost (1x1 array)
        '''

        # Sometimes L-BFGS-B was crazy - flattening the vector
        theta = theta.flatten()

        # first element is the amplitude
        self.width = theta[0]

        # the remaining elements are the characteristic lengthscales
        self.scale = theta[1:]

        # compute alpha
        alpha_ = self.alpha()

        # trick to compute the determinant once we have
        # already computed the Cholesky factor
        det_ = np.log(np.diag(self.L)).sum(0)

        # compute the cost
        cst = 0.5 * (self.y * alpha_).sum(0) + det_

        return cst

    def grad_log_like(self, theta):
        '''
        Function to calculate the gradient of the cost
        (negative log-marginal likelihood) with respect to
        the kernel hyperparameters

        Args:
            (array) theta: the kernel hyperparameters in
                            the correct order

        Returns:
            (array) gradient: vector of the gradient
        '''

        # the kernel hyperparameters
        theta = theta.flatten()

        # amplitude
        self.width = theta[0]

        # characteristic lengthscales
        self.scale = theta[1:]

        # Number of parameters
        Npar = len(theta)

        # empty array to record the gradient
        gradient = np.zeros(Npar)

        # compute alpha
        alpha_ = self.alpha()

        # compute K^-1 via triangular method
        kinv = gpl.dpotrs(self.L, np.eye(self.N), lower=True)[0]

        # see expression for gradient
        A = np.einsum('i,j', alpha_.flatten(), alpha_.flatten()) - kinv

        # Gradient calculation with respect
        # to hyperparameters (hard-coded)
        grad = {}
        K_rbf = self.rbf('trainSet', self.theta_, self.theta_)

        grad['0'] = 2.0 * K_rbf
        for i in range(self.ndim):
            dist_ = self.distance_per_dim(self.theta_[:, i], self.theta_[:, i])
            grad[str(i + 1)] = K_rbf * dist_ / np.exp(2.0 * self.scale[i])

        for i in range(Npar):
            gradient[i] = 0.5 * gpl.trace_dot(A, grad[str(i)])

        return -gradient

    def fit(self, method='CG', bounds=None, options={'ftol': 1E-5}):
        '''
        Function to do the optimisation (training)

        Args:
            (str) method: optimisation method from scipy
                    see scipy.optimize.minimize for further details

            (array) bounds: some methods also allow for a bound/prior

            (dict) options: can also pass convergence conditions etc...
        '''

        bounds_ = np.array(bounds)

        # empty list to record the cost
        minChi2 = []

        # empty list to record the optimum parameters
        recordParams = []

        # if we want to train
        if self.train:

            for i in range(self.nrestart):

                print('Performing Optimization step {}'.format(i + 1))

                # an initial guess from the bound/prior
                myGuess = np.random.uniform(bounds_[:, 0], bounds_[:, 1])

                # optimisation!
                soln = op.minimize(
                    self.cost, myGuess, method=method,
                    bounds=bounds, jac=self.grad_log_like, options=options)

                # record optimised solution (cost)
                minChi2.append(np.ones(1) * soln.fun)

                # record optimum parameters
                recordParams.append(soln.x)

            # just converting list to arrays
            self.minChi2 = np.array(minChi2).reshape(self.nrestart,)
            self.recordParams = np.array(recordParams)

            # sometimes we found crazy behaviour
            # maybe due to numerical errors
            # ignore NaN in cost
            if np.isnan(self.minChi2).any():
                index = np.argwhere(np.isnan(self.minChi2))
                self.minChi2 = np.delete(self.minChi2, index)
                self.recordParams = np.delete(self.recordParams, index, axis=0)

            self.logger.info(
                'The value of the cost is {}'.format(self.minChi2))

            # choose the hyperparameters with minimum cost
            cond = self.minChi2 == np.min(self.minChi2)
            optParams = self.recordParams[cond][0]
            optParams = optParams.flatten()

            self.logger.info('Optimum is {}'.format(optParams))

            # update amplitude
            self.width = optParams[0]

            # update characteristic lengthscales
            self.scale = optParams[1:]

            # Update alpha (after training) - no need to update kernel
            # because already updated in function cost
            self.alpha_ = self.alpha()

        else:
            self.alpha_ = self.alpha()

        return None

    def prediction(self, test_point, returnVar=True):
        '''
        Function to make predictions given a test point

        Args:
            (array) test_point: a test point of length ndim

            (bool) returnVar: If True, the GP variance will
                    be computed

        Returns:
            (array) mean, var: if returnVar=True
            (array) mean : if returnVar=False
        '''

        # use numpy array instead of list (if any)
        test_point = np.array(test_point).flatten()

        assert len(test_point) == self.ndim, 'Different dimension'

        # transform point first
        test_point_trans = np.dot(self.MU, test_point)
        test_point_trans = test_point_trans.reshape(1, self.d)

        # compute the k_star vector
        ks = self.kernel('trainTest', self.theta_, test_point_trans)

        # compute mean GP - super quick
        meanGP = np.array([(ks.flatten() * self.alpha_.flatten()).sum(0)])

        # rescale back
        mu = self.mean_y + self.std_y * meanGP

        # do extra computations if we want GP variance
        if returnVar:
            v = gpl.dpotrs(self.L, ks, lower=True)[0].flatten()
            kss = self.kernel('testSet', test_point_trans, test_point_trans)
            varGP = kss - (ks.flatten() * v).sum(0)
            varGP = varGP.flatten()

            # rescale back
            var = self.std_y**2 * varGP
            return mu, var
        else:
            return mu

    def sampleBandPower(self, test_point, mean=True, nsamples=200):
        '''
        Function to generate samples of the original
        MOPED coefficient given a test point

        This is important if we want to marginalise over the GP
        uncertainty numerically.

        Args:
            (array) test_point: an array of size ndim

            (bool) mean: if True, only the mean will be computed

            (int) nsamples: if specified, n samples will be drawn

        Returns:
            (array) mean: mean value of GP of size 1

            (array) samples: samples from the GP of size nsamples
        '''

        # compute mean
        if mean:
            mu = self.prediction(test_point, returnVar=False)
            return np.power(10, mu) - self.constant

        # or return samples
        else:
            mu, var = self.prediction(test_point, returnVar=True)
            samples = np.random.normal(mu, np.sqrt(var), nsamples)
            return np.power(10, samples) - self.constant
