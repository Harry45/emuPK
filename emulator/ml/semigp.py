'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Semi-Parametric Gaussian Process
'''


import sys
from collections import OrderedDict
from functools import reduce
import numpy as np
import scipy.optimize as op
import scipy.linalg as sl
from GPy.util import linalg as gpl
from emulator.utils.kernel import rbf, squared_distance
from emulator.utils.algebra import solve, matrix_inverse, diagonal
from emulator.utils.transformation import transformation


class GP:

    '''
    Inputs
    ------
    theta (np.ndarray) : matrix of size ntrain x ndim

    y (np.ndarray) : output/target

    var (float or np.ndarray) : noise covariance matrix of size ntrain x ntrain

    x_trans (bool) : if True, pre-whitening is applied

    y_trans (bool) : if True, log of output is used

    jitter (float) : a jitter term just to make sure all matrices are numerically stable

    use_mean (bool) : if True, the outputs are centred on zero
    '''

    def __init__(self, theta, y, var=1E-5, order = 2, x_trans=False, y_trans=False, jitter=1E-10, use_mean=False):

        # compute mean of training set
        self.mean_theta = np.mean(theta, axis=0)

        # the jitter term for numerical stability
        self.jitter = jitter

        # centre the input on zero
        self.theta = theta - self.mean_theta

        msg = 'The number of training points is smaller than the dimension of the problem. Reshape your array!'
        # the number of training points is greater than the number of dimension
        assert self.theta.shape[0] > self.theta.shape[1], msg

        # the dimension of the problem
        self.d = self.theta.shape[1]

        # the number of training point
        self.ntrain = self.theta.shape[0]

        # if we want to centre the output on zero
        self.use_mean = use_mean

        if self.use_mean:
            self.mean_function = np.mean(y)
        else:
            self.mean_function = np.zeros(1)

        # the output is of size (ntrain x 1)
        self.y = y.reshape(self.ntrain, 1) - self.mean_function

        # noise ccovariance matrix
        self.var = np.atleast_2d(var)

        # choose to make transformation
        self.x_trans = x_trans
        self.y_trans = y_trans

        # store kernel hyperparameters
        self.opt_params = None

        # order of the poynomial regression
        # we support only second order here
        self.order = order
        if self.order > 2:
            msg =  'At the moment, we support only order = 1 and order = 2'
            raise RuntimeError(msg)

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
        test_point (np.ndarray) : if a test point is provdied, phi_star is calculated

        Returns
        -------
        phi or phi_star (np.ndarray) : the basis functions
        '''

        # we need to make the transformation first
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

            if (isinstance(mean, np.ndarray) and isinstance(cov, np.ndarray)):

                msg = 'The shape of the prior does not match with the shape of the design matrix'
                assert len(mean) == self.nbasis, msg
                assert cov.shape[0] == cov.shape[1] == self.nbasis, msg

                self.mu = mean.reshape(self.nbasis, 1)
                self.cov = lambda_cap * cov

            # we assign zero mean and unit variance (times lambda_cap) by default
            elif (mean is None and cov is None):
                self.mu = np.zeros((self.nbasis, 1))
                self.cov = lambda_cap * np.identity(self.nbasis)

            # Compute difference between the output vector and the polynomial part
            self.diff = self.y_train - np.dot(self.phi, self.mu)

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

        # calculate inverse
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
            raise RuntimeError(
                'Input the priors for the regression coefficients first. See function regression_prior!')

        else:

            mat_inv = matrix_inverse(self.cov, return_chol=False)

            return mat_inv

    def posterior(self):
        '''
        Computes the posterior distribution of beta and f (latent variables)

        Note: Optimise for the kernel parameters first

        Inputs
        ------

        Returns
        -------
        post_mean (np.ndarray) : mean posterior

        a_inv_matrix (np.ndarray) : covariance of all latent parameters
        '''

        # Compute the noise_ccovariance matrix
        noise_cov = self.noise_covariance()

        # Compute gamma
        gamma = np.vstack([self.mu, np.zeros((self.ntrain, 1))])

        # Compute the matrix D
        D = np.c_[self.phi, np.eye(self.ntrain)]

        # Compute the kernel matrix - assuming we already have the optimised parameters
        K = rbf(x_train=self.x_train, params=np.exp(self.opt_params))

        # Compute matrix R and its inverse
        R = sl.block_diag(self.cov, K + np.identity(self.ntrain) * self.jitter)
        Rinv = matrix_inverse(R, return_chol=False)

        # Compute A and b
        A = np.dot(D.T, solve(noise_cov, D)) + Rinv
        b = np.dot(D.T, solve(noise_cov, self.y_train)) + np.dot(Rinv, gamma)

        # Compute a_inv_matrix - this is the covariance of all latent variables
        a_inv_matrix = matrix_inverse(A, return_chol=False)

        # Mean posterior
        post_mean = np.dot(a_inv_matrix, b)

        return post_mean, a_inv_matrix

    def evidence(self, params):
        '''
        Calculate the log-evidence of the GP

        Inputs
        ------
        params (np.ndarray) : kernel hyperparameters

        Outputs
        -------
        neg_log_evidence (np.ndarray) : the negative log-marginal likelihood

        -gradient (np.ndarray) : the gradient with respect to the kernel hyperparameters
        '''

        # sometimes the optimizer prefers a 1D array!
        params = params.flatten()

        # number of hyperparameters
        n_par = len(params)

        # empty array to store the gradient
        gradient = np.zeros(n_par)

        # Pre-compute some quantities
        # The second one can be computed only once but it is of size N x N (we
        # do not want to store it in memory)
        noise_cov = self.noise_covariance()
        basis = reduce(np.dot, [self.phi, self.cov, self.phi.T])
        kernel = rbf(x_train=self.x_train, params=np.exp(params))

        # Sum of the above
        total_kernel = noise_cov + basis + kernel

        # Compute eta
        eta, chol_factor = solve(total_kernel, self.diff, return_chol=True)

        # Compute k_inv
        k_inv = gpl.dpotrs(chol_factor, np.eye(chol_factor.shape[0]), lower=True)[0]

        # The term in the bracket in the gradient expression
        bracket_term = np.einsum('i,j', eta.flatten(), eta.flatten()) - k_inv

        # Gradient calculation with respect to hyperparameters (hard-coded)
        grad = {}
        grad['0'] = kernel

        for i in range(1, n_par):
            train_points = np.atleast_2d(self.x_train[:, i - 1]).T
            grad[str(i)] = kernel * squared_distance(train_points, train_points, np.exp(params[i]))

        for i in range(n_par):
            gradient[i] = gpl.trace_dot(bracket_term, grad[str(i)])

        # compute log-evidence

        # the determinant part
        det = np.log(np.diag(chol_factor)).sum(0)

        # the fitting (chi-squared) term
        cost = 0.5 * (self.diff * eta).sum(0)

        # the constant term (irrelevant but let's keep it)
        cnt = 0.5 * self.ntrain * np.log(2.0 * np.pi)

        # the total log-marginal likelihood
        neg_log_evidence = cost + det + cnt

        return neg_log_evidence, -gradient

    def fit(self, method='CG', bounds=None, options={'ftol': 1E-5}, n_restart=2):
        '''
        The kernel hyperparameters are learnt in this function.

        Inputs
        ------
        method (str) : the choice of the optimizer:

            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

            Recommend L-BFGS-B algorithm

        bounds (np.ndarray) : the prior on these hyperparameters

        options (dictionary) : options for the L-BFGS-B optimizer. We have:

            options={'disp': None,
                    'maxcor': 10,
                    'ftol': 2.220446049250313e-09,
                    'gtol': 1e-05,
                    'eps': 1e-08,
                    'maxfun': 15000,
                    'maxiter': 15000,
                    'iprint': - 1,
                    'maxls': 20,
                    'finite_diff_rel_step': None}

        n_restart (int) : number of times we want to restart the optimizer

        Returns
        -------
        opt_params (np.ndarray) : array of the optimised kernel hyperparameters
        '''

        # make sure the bounds are arrays
        bounds_ = np.array(bounds)

        # empty list to store the minimum chi_square
        min_chi_sqr = []

        # empty list to store the optimised parameters
        record_params = []

        for i in range(n_restart):

            print('Performing Optimization step {}'.format(i + 1))

            # take a guess between the bounds provided
            guess = np.random.uniform(bounds_[:, 0], bounds_[:, 1])

            # perform the optimisation
            soln = op.minimize(self.evidence, guess, method=method, bounds=bounds, jac=True, options=options)

            min_chi_sqr.append(np.ones(1) * soln.fun)
            record_params.append(soln.x)

        # store the minimum chi_square
        self.min_chi_sqr = np.array(min_chi_sqr).reshape(n_restart,)

        # store the optimised parameters
        self.record_params = np.array(record_params)

        # sometimes the optimiser returns NaN - ignore these
        if np.isnan(self.min_chi_sqr).any():
            index = np.argwhere(np.isnan(self.min_chi_sqr))
            self.min_chi_sqr = np.delete(self.min_chi_sqr, index)
            self.record_params = np.delete(self.record_params, index, axis=0)

        print('{}'.format(self.min_chi_sqr))

        opt_params = self.record_params[self.min_chi_sqr == np.min(self.min_chi_sqr)][0]
        opt_params = opt_params.flatten()

        print('{}'.format(opt_params))

        self.opt_params = opt_params

        # pre-compute some important quantities once we optimise for the kernel
        # hyperparameters
        noise_cov = self.noise_covariance()
        kernel = rbf(x_train=self.x_train, params=np.exp(self.opt_params))
        kernel_y = noise_cov + kernel

        # compute posterior mean and variance of model
        m, c = self.posterior()
        self.beta_hat = m[0:-self.ntrain]
        self.var_beta = c[0:self.nbasis, 0:self.nbasis]
        del c
        self.alpha_1 = solve(kernel_y, self.phi, return_chol=False)
        self.alpha_2, self.chol_stored = solve(kernel_y, self.y_train, return_chol=True)

        return opt_params

    def prediction(self, test_point, return_var=False):
        '''
        Predicts the function at a test point in parameter space

        Inputs
        ------
        test_point (np.ndarray) : test point in parameter space

        return_var (bool) : if True, the predicted variance will be computed

        Returns
        -------
        mean_pred (np.ndarray) : the mean of the GP

        var_pred (np.ndarray) : the variance of the GP (optional)
        '''

        # transform point first
        test_point = np.array(test_point).flatten() - self.mean_theta

        if self.x_trans:
            test_point = self.transform.x_transform_test(test_point)

        ks, kss = rbf(x_train=self.x_train, x_test=test_point, params=np.exp(self.opt_params))

        # compute basis at test_point
        phi_star = self.compute_basis(test_point=test_point)

        # compute x_star
        x_star = phi_star - np.dot(ks.T, self.alpha_1)
        f_star = np.dot(ks.T, self.alpha_2)

        # mean prediction
        pred = np.dot(x_star, self.beta_hat) + f_star

        mean_pred = pred + self.mean_function

        # Compute variance
        if return_var:
            term1 = reduce(np.dot, [x_star, self.var_beta, x_star.T]).flatten()
            term2 = (ks.flatten() * gpl.dpotrs(self.chol_stored, ks, lower=True)[0].flatten()).sum(0)
            var_pred = term1 + self.var + kss - term2
            return mean_pred, var_pred

        else:
            return mean_pred

    def pred_original_function(self, test_point, n_samples=None):

        if not self.y_trans:
            msg = 'You must transform the target in order to use this function'
            raise RuntimeWarning(msg)

        if n_samples:
            mu, var = self.prediction(test_point, return_var=True)
            samples = np.random.normal(mu.flatten(), np.sqrt(var).flatten(), n_samples)
            ySamples = self.transform.y_inv_transform_test(samples)
            return ySamples
        else:
            mu = self.prediction(test_point, return_var=False)
            y_original = self.transform.y_inv_transform_test(mu)
            return y_original

    def grad_pre_computations(self, test_point):
        '''
        Pre-compute some quantities prior to calculating the gradients

        Inputs
        ------
        test_point (np.ndarray) : test point in parameter space

        Returns
        -------
        grad (np.ndarray) : first derivative

        gradient_sec (np.ndarray) : second derivative

        '''

        # transform test point
        test_point = np.array(test_point).flatten() - self.mean_theta

        # make a copy of original test point
        test_point_ = np.copy(test_point)

        # we need the MU matrix to compute the gradient
        if self.x_trans:
            test_point = self.transform.x_transform_test(test_point)
            mu_matrix = self.transform.mu_matrix
        else:
            mu_matrix = np.identity(self.d)

        # we also need k_star
        ks, kss = rbf(x_train=self.x_train, x_test=test_point, params=np.exp(self.opt_params))
        del kss

        if len(self.opt_params[1:]) == 1:
            diag = np.diag(1. / np.exp(2.0 * np.repeat(self.opt_params[1:], self.d)))

        else:
            diag = np.diag(1. / np.exp(2.0 * self.opt_params[1:]))

        Q = np.dot(mu_matrix.T, np.dot(np.atleast_2d(diag), mu_matrix))
        diff = self.theta - test_point_.reshape(1, self.d)
        alpha_ = self.alpha_2 - np.dot(self.alpha_1, self.beta_hat)

        dk_dtheta = ks * np.dot(diff, Q)
        gradient_first = np.dot(dk_dtheta.T, alpha_)

        # there are multiple conditions here
        # order = 1 and we require first derivative
        # order = 2 and we require first derivative
        # order = 1 and we require first and second derivatives
        # order = 2 and we require first and second derivatives
        if self.order == 1:
            grad = np.dot(mu_matrix.T, self.beta_hat[1:]) + gradient_first

        elif self.order == 2:
            to_add = np.zeros((1, self.d))
            for i in range(self.d):
                to_add += self.beta_hat[-self.d + i] * 2 * reduce(np.dot, [np.atleast_2d(
                    mu_matrix[i]), np.atleast_2d(test_point_).T, np.atleast_2d(mu_matrix[i])])
            grad = np.dot(
                mu_matrix.T, self.beta_hat[1:self.d + 1]).flatten() + gradient_first.flatten() + to_add.flatten()

        grad = grad.flatten()

        # second derivative
        dummy_ = dk_dtheta * np.dot(diff, Q)
        gradient_sec = -Q * np.dot(ks.T, alpha_) + np.dot(dummy_.T, alpha_)
        gradient_sec = np.diag(gradient_sec)

        if self.order == 2:

            rec = []

            for i in range(self.d):
                rec.append(2 * self.beta_hat[-self.d + i].flatten() * mu_matrix[i]**2)

            # this requires the gradient_sec (from order = 1)
            gradient_sec = gradient_sec + np.array(rec).sum(axis=0)

        return grad, gradient_sec

    def derivatives(self, test_point, order=1):
        '''
        If we did some transformation on the ouputs, we need this function to calculate
        the 'exact' gradient

        Inputs
        ------
        test_point (np.ndarray) : array of the test point

        order (int) : 1 or 2, referrring to first and second derivatives respectively

        Returns
        -------
        grad (np.ndarray) : first derivative with respect to the input parameters

        gradient_sec (np.ndarray) : second derivatives with respect to the input parameters

        '''

        # make a copy of original test point
        test_point_ = np.copy(test_point)

        test_point = np.array(test_point).flatten()

        grad, gradient_sec = self.grad_pre_computations(test_point)

        if (self.x_trans and self.y_trans):

            mu = self.prediction(test_point_, return_var=False)

            # this is due to the log_10 transformation - check derivation again
            grad_y = 10**mu * np.log(10) * grad
            grad_y = grad_y.flatten()

            if order == 1:
                return grad_y

            else:
                # we need to check this again for second derivatives
                grad2_y = (grad.flatten() + 10**mu.flatten()
                           * gradient_sec) * np.log(10)
                return grad_y, grad2_y

        else:
            if order == 1:
                return grad
            else:
                return grad, gradient_sec
