'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Routine for polynomial regression
'''

from functools import reduce
import scipy.linalg as sl
import numpy as np
from emulator.utils.algebra import solve, matrix_inverse, diagonal
from emulator.utils.transformation import transformation


class GLM:

    '''
    Gaussian Linear Model (GLM) class for polynomial regression

    Inputs
    ------

    theta (np.ndarray) : matrix of size ntrain x ndim

    y (np.ndarray) : output/target

    var (float or np.ndarray) : noise covariance matrix of size ntrain x ntrain

    order (int) : order of polynomial regression

    x_trans (bool) : if True, pre-whitening is applied

    y_trans (bool) : if True, log of output is used

    use_mean (bool) : if True, the outputs are centred on zero
    '''

    def __init__(self, theta, y, order=2, var=1E-5, x_trans=False, y_trans=False, use_mean=True):

        # calculate mean inputs
        self.mean_theta = np.mean(theta, axis=0)

        # scale the inputs to zero
        self.theta = theta - self.mean_theta

        msg = 'The number of training points is smaller than the dimension of the problem. Reshape your array!'

        assert self.theta.shape[0] > self.theta.shape[1], msg

        # dimension of the problem
        self.d = self.theta.shape[1]

        # number of training points
        self.ntrain = self.theta.shape[0]

        # use mean function or not
        self.use_mean = use_mean

        if self.use_mean:
            self.mean_function = np.mean(y)
        else:
            self.mean_function = np.zeros(1)

        # scale output to zero depending on whether use_mean is set to True
        self.y = y.reshape(self.ntrain, 1) - self.mean_function

        # noise covariance matrix
        self.var = np.atleast_2d(var)

        # do we want to make transformation
        self.x_trans = x_trans
        self.y_trans = y_trans

        # order of polynomial
        self.order = order

    def do_transformation(self):
        '''
        Perform all transformations

        Inputs
        ------

            None

        Outputs
        -------

            None
        '''

        # we transform both x and y if specified
        if (self.x_trans and self.y_trans):
            self.transform = transformation(self.theta, self.y)
            self.x_train = self.transform.x_transform()
            self.y_train = self.transform.y_transform()

        # we transform x only if specified
        elif self.x_trans:
            self.transform = transformation(self.theta, self.y)
            self.x_train = self.transform.x_transform()
            self.y_train = self.y

        # we keep the inputs and outputs (original basis)
        else:
            self.x_train = self.theta
            self.y_train = self.y

    def compute_basis(self, test_point=None):
        '''
        Compute the input basis functions

        Inputs
        ------

        test_point (np.ndarray: optional) : if a test point is provided, phi_star is calculated

        Returns
        -------

        phi or phi_star (np.ndarray) : the basis functions
        '''

        if test_point is None:

            if not hasattr(self, "x_train"):
                raise RuntimeError('Make the appropriate transformation first')

            else:
                dummy_phi = [self.x_train**i for i in np.arange(1, self.order + 1)]
                self.phi = np.concatenate(dummy_phi, axis=1)
                self.phi = np.c_[np.ones((self.x_train.shape[0], 1)), self.phi]
                self.nbasis = self.phi.shape[1]

            return self.phi

        else:
            dummy_phi_star = np.array([test_point**i for i in np.arange(1, self.order + 1)]).flatten()
            phi_star = np.c_[np.ones((1, 1)), np.atleast_2d(dummy_phi_star)]

            return phi_star

    def regression_prior(self, mean=None, cov=None, lambda_cap=1):
        '''
        Specify the regression prior (mean and covariance)

        Inputs
        ------

        mean (np.ndarray) : default zeros

        cov (np.ndarray) : default identity matrix

        lambda_cap (float) : width of the prior covariance matrix (default 1)

        Returns
        -------

        '''

        # compute the design matrix first
        if not hasattr(self, 'phi'):
            raise RuntimeError('Compute the design matrix first')

        else:

            if (mean is not None and cov is not None):
                msg = 'The shape of the prior does not match with the shape of the design matrix'
                assert len(mean) == self.nbasis, msg
                assert cov.shape[0] == cov.shape[1] == self.nbasis, msg

                self.mu = mean.reshape(self.nbasis, 1)
                self.cov = cov

            # we assign zero mean and unit variance (times lambda_cap) by default
            elif (mean is None and cov is None):
                self.mu = np.zeros((self.nbasis, 1))
                self.cov = lambda_cap * np.identity(self.nbasis)

    def noise_covariance(self):
        '''
        Build the noise covariance matrix

        Inputs
        ------

        Returns
        -------
        '''

        if (self.var.shape[0] == self.var.shape[1] == self.ntrain):
            return self.var
        else:
            return self.var * np.identity(self.ntrain)

    def inv_noise_cov(self):
        '''
        Calculate the inverse of the noise covariance matrix

        Inputs
        ------

        Returns
        -------

        mat_inv (np.ndarray) : inverse of the noise covariance

        '''

        # Compute noise covariance first
        noise_cov = self.noise_covariance()

        mat_inv = matrix_inverse(noise_cov, return_chol=False)

        return mat_inv

    def inv_prior_cov(self):
        '''
        Calculate the inverse of the prior covariance matrix

        Inputs
        ------

        Returns
        -------

        mat_inv (np.ndarray) : inverse of the prior covariance matrix (parametric part)
        '''

        if not hasattr(self, 'cov'):
            msg = 'Input the priors for the regression coefficients first. See function regression_prior!'
            raise RuntimeError(msg)

        else:
            mat_inv = matrix_inverse(self.cov, return_chol=False)

            return mat_inv

    def evidence(self):
        '''
        Calculates the log-evidence of the model

        Inputs
        ------

        Returns
        -------

        log_evidence (np.ndarray) : the log evidence of the model

        '''

        diff = self.y_train - np.dot(self.phi, self.mu)
        noise_cov = self.noise_covariance()
        new_cov = noise_cov + reduce(np.dot, [self.phi, self.cov, self.phi.T])

        dummy, chol_factor = solve(new_cov, diff, return_chol=True)

        # compute the log-evidence

        # determinant term
        det = 2.0 * np.log(np.diag(chol_factor)).sum()

        # constant term
        cnt = self.ntrain * np.log(2.0 * np.pi)

        # fit term
        fit_term = np.dot(diff.T, dummy)

        log_evidence = -0.5 * (fit_term + det + cnt)

        print('The log-evidence is {:.2f}'.format(float(log_evidence)).center(50))

        return log_evidence

    def posterior_coefficients(self):
        '''
        Calculate the posterior coefficients

        Inputs
        ------

        Returns
        -------

        beta_bar (np.ndarray) : mean posterior

        lambda_cap (np.ndarray) : covariance of the regression coefficients
        '''

        # Compute noise covariance
        noise_cov = self.noise_covariance()

        # Compute inverse of the prior covariance
        cov_inv = self.inv_prior_cov()

        # Compute covariance of regression coefficients
        lambda_cap = cov_inv + np.dot(self.phi.T, solve(noise_cov, self.phi))
        lambda_cap = matrix_inverse(lambda_cap, return_chol=False)

        # Compute mean of regression coefficients
        dummy = np.dot(self.phi.T, solve(noise_cov, self.y_train)) + np.dot(cov_inv, self.mu)
        beta_bar = np.dot(lambda_cap, dummy)

        # store the mean and covariance of the posterior
        # in order to calculate predictions at test point
        # in the parameter space
        self.beta_bar = beta_bar
        self.lambda_cap = lambda_cap

        return beta_bar, lambda_cap

    def prediction(self, test_point):
        '''
        Given a test point, the prediction (mean and variance) will be computed

        TODO: need to improve this function to account for more than 1 dimension

        Inputs
        ------

        test_point (np.ndarray) : vector of test point in parameter space

        Returns
        -------

        post_mean (np.ndarray) : mean of the posterior

        post_var (np.ndarray) : variance of the posterior
        '''

        # scale the test point according to the training point
        test_point = np.atleast_2d(test_point - self.mean_theta)

        msg = 'Dimension of test point is not the same as training points'

        assert test_point.shape[1] == self.d, msg

        # Compute phi_star
        if test_point.shape[1] == 1:
            phi_star = self.compute_basis(test_point=test_point)

        elif self.x_trans:
            # transform test point first
            test_point_trans = self.transform.x_transform_test(test_point)

            phi_star = self.compute_basis(test_point=test_point_trans)

        else:
            phi_star = self.compute_basis(test_point=test_point)

        # Compute mean and variance
        post_mean = np.dot(phi_star, self.beta_bar)
        post_var = reduce(np.dot, [phi_star, self.lambda_cap, phi_star.T])

        # if we have a fixed noise term, this gets added
        # to the posterior predictive function

        if self.var.shape[0] == 1:
            post_var += self.var

        post_mean = post_mean.flatten()

        post_var = post_var.flatten()

        return post_mean, post_var
