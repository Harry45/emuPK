# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Zero Mean Gaussian Process
'''

from typing import Tuple
import numpy as np
import scipy.optimize as op
from GPy.util import linalg as gpl

from ml.kernel import rbf, squared_distance
from ml.algebra import solve
from ml.transformation import transformation


class GP:

    '''
    Module to perform a zero mean Gaussian Process regression. One can also specify if we want to apply
    the pre-whitening step at the input level and the logarithm transformation at the output level.

    :param: theta (np.ndarray) : matrix of size ntrain x ndim

    :param: y (np.ndarray) : output/target

    :param: var (float or np.ndarray) : noise covariance matrix of size ntrain x ntrain

    :param: x_trans (bool) : if True, pre-whitening is applied

    :param: y_trans (bool) : if True, log of output is used

    :param: use_mean (bool) : if True, the outputs are centred on zero
    '''

    def __init__(
            self,
            theta: np.ndarray,
            y: np.ndarray,
            var: float = 1E-5,
            x_trans: bool = False,
            y_trans: bool = False,
            use_mean: bool = False):

        # compute mean of training set
        self.mean_theta = np.mean(theta, axis=0)

        # centre the input on zero
        self.theta = theta - self.mean_theta

        msg = 'The number of training points is smaller than the dimension of the problem. Reshape your array!'
        # the number of training points is greater than the number of dimension
        assert self.theta.shape[0] > self.theta.shape[1], msg

        # the dimension of the problem
        self.d = self.theta.shape[1]

        # the number of training point
        self.ntrain = self.theta.shape[0]

        # the targets
        self.y = y

        # noise ccovariance matrix
        self.var = np.atleast_2d(var)

        # x-transformation
        self.x_trans = x_trans

        # y-transfrmation
        self.y_trans = y_trans

        # do we want to scale the training points on zero
        self.use_mean = use_mean

        if self.use_mean:
            self.mean_function = np.mean(y)
        else:
            self.mean_function = np.zeros(1)

        # the output is of size (ntrain x 1)
        self.y = y.reshape(self.ntrain, 1) - self.mean_function

    def do_transformation(self) -> None:
        '''
        Perform all transformations
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

    def noise_covariance(self) -> np.ndarray:
        '''
        Build the noise covariance matrix

        :return: the pre-defined (co-)variance in its appropriate form
        '''

        if (self.var.shape[0] == self.var.shape[1] == self.ntrain):
            return self.var
        else:
            return self.var * np.identity(self.ntrain)

    def evidence(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate the log-evidence of the GP and the gradient with respect to the kernel hyperparameters

        :param: params (np.ndarray) : kernel hyperparameters

        :return: neg_log_evidence (np.ndarray) : the negative log-marginal likelihood

        :return: -gradient (np.ndarray) : the gradient with respect to the kernel hyperparameters
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
        kernel = rbf(x_train=self.x_train, params=np.exp(params))

        # Sum of the above
        total_kernel = noise_cov + kernel

        # Compute eta
        eta, chol_factor = solve(total_kernel, self.y_train, return_chol=True)

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
        cost = 0.5 * (self.y_train * eta).sum(0)

        # the constant term (irrelevant but let's keep it)
        cnt = 0.5 * self.ntrain * np.log(2.0 * np.pi)

        # the total log-marginal likelihood
        neg_log_evidence = cost + det + cnt

        return neg_log_evidence, -gradient

    def fit(
            self,
            method: str = 'CG',
            bounds: np.ndarray = None,
            options: dict = {
                'ftol': 1E-5},
            n_restart: int = 2) -> np.ndarray:
        '''
        The kernel hyperparameters are learnt in this function.

        :param: method (str) : the choice of the optimizer:

            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

            Recommend L-BFGS-B algorithm

        :param: bounds (np.ndarray) : the prior on these hyperparameters

        :param: options (dictionary) : options for the L-BFGS-B optimizer. We have:

        .. code-block:: python

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

        :param: n_restart (int) : number of times we want to restart the optimizer

        :return: opt_params (np.ndarray) : array of the optimised kernel hyperparameters
        '''

        # make sure the bounds are arrays
        bounds_ = np.array(bounds)

        # empty list to store the minimum chi_square
        min_chi_sqr = []

        # empty list to store the optimised parameters
        record_params = []

        for i in range(n_restart):

            # print('Performing Optimization step {}'.format(i + 1))

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

        # print('{}'.format(opt_params))

        self.opt_params = opt_params

        # pre-compute some important quantities once we optimise for the kernel
        # hyperparameters
        noise_cov = self.noise_covariance()
        kernel = rbf(x_train=self.x_train, params=np.exp(self.opt_params))
        kernel_y = noise_cov + kernel

        # compute posterior mean and variance of model
        self.alpha, self.chol_stored = solve(kernel_y, self.y_train, return_chol=True)

        return opt_params

    def prediction(self, test_point: np.ndarray, return_var: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Predicts the function at a test point in parameter space

        :param: test_point (np.ndarray) : test point in parameter space

        :param: return_var (bool) : if True, the predicted variance will be computed

        :return: mean_pred (np.ndarray) : the mean of the GP

        :return: var_pred (np.ndarray) : the variance of the GP (optional)
        '''

        # transform point first
        test_point = np.array(test_point).flatten() - self.mean_theta

        if self.x_trans:
            test_point = self.transform.x_transform_test(test_point)

        ks, kss = rbf(x_train=self.x_train, x_test=test_point, params=np.exp(self.opt_params))

        pred = np.dot(ks.T, self.alpha)

        mean_pred = pred + self.mean_function

        mean_pred = mean_pred.reshape(1,)

        # Compute variance
        if return_var:
            term = (ks.flatten() * gpl.dpotrs(self.chol_stored, ks, lower=True)[0].flatten()).sum(0)
            var_pred = self.var + kss - term
            return mean_pred, var_pred

        else:
            return mean_pred

    def pred_original_function(self, test_point: np.ndarray, n_samples: int = None) -> np.ndarray:
        '''
        Calculates the original function if the log_10 transformation is used on the target.

        :param: test_point (np.ndarray) - the test point in parameter space

        :param: n_samples (int) - we can also generate samples of the function, assuming we have stored the Cholesky factor

        :return: y_samples (np.ndarray) - if n_samples is specified, samples will be returned

        :return: y_original (np.ndarray) - the predicted function in the linear scale (original space) is returned
        '''

        if not self.y_trans:
            msg = 'You must transform the target in order to use this function'
            raise RuntimeWarning(msg)

        if n_samples:
            mu, var = self.prediction(test_point, return_var=True)
            samples = np.random.normal(mu.flatten(), np.sqrt(var).flatten(), n_samples)
            y_samples = self.transform.y_inv_transform_test(samples)
            return y_samples
        else:
            mu = self.prediction(test_point, return_var=False)
            y_original = self.transform.y_inv_transform_test(mu)
            return y_original

    def grad_pre_computations(self, test_point: np.ndarray, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Pre-compute some quantities prior to calculating the gradients

        :param: test_point (np.ndarray) : test point in parameter space

        :param: order (int) : order of differentiation (default: 1) - not to be confused with order of the polynomial

        :return: gradients (tuple) : first and second derivatives (if order = 2)
        '''

        # transform test point
        tp = np.array(test_point).flatten() - self.mean_theta

        # we need the transformation matrix to compute the gradient
        if self.x_trans:
            tp_trans = self.transform.x_transform_test(tp)
            mu_mat = self.transform.mu_matrix
        else:
            tp_trans = tp
            mu_mat = np.identity(self.d)

        # we also need k_star
        ks, _ = rbf(x_train=self.x_train, x_test=tp_trans, params=np.exp(self.opt_params))

        if len(self.opt_params[1:]) == 1:
            diag = np.diag(1. / np.exp(2.0 * np.repeat(self.opt_params[1:], self.d)))

        else:
            diag = np.diag(1. / np.exp(2.0 * self.opt_params[1:]))

        # this is Omega^-1 in notes
        Q = np.dot(mu_mat.T, np.dot(np.atleast_2d(diag), mu_mat))

        # this is Z in notes
        diff = self.theta - tp.reshape(1, self.d)

        # difference dot with correlation matrix
        D_dot_Q = np.dot(diff, Q)

        # derivative of the kernel with respect to the input
        dk_dtheta = ks * D_dot_Q

        # gradient of the residual
        gradient_first = np.dot(dk_dtheta.T, self.alpha).flatten()

        if order == 1:
            return gradient_first, None

        elif order == 2:

            # empty array to record the second derivatives
            gradient_sec = np.zeros((self.d, self.d))

            for i in range(self.d):
                for j in range(self.d):
                    gradient_sec[i, j] = sum(dk_dtheta[:, i] * D_dot_Q[:, j] * self.alpha.flatten())

            gradient_sec += -Q * np.dot(ks.T, self.alpha)

            return gradient_first, gradient_sec

        else:
            ValueError('Only Order 1 and Order 2 supported!')

    def derivatives(self, test_point: np.ndarray, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        '''
        If we did some transformation on the ouputs, we need this function to calculate the 'exact' gradient

        :param: test_point (np.ndarray) : array of the test point

        :param: order (int) : 1 or 2, referrring to first and second derivatives respectively


        :return: grad (np.ndarray) : first derivative with respect to the input parameters

        :return: gradient_sec (np.ndarray) : second derivatives with respect to the input parameters, if specified
        '''

        # make a copy of original test point
        test_point_ = np.copy(test_point)

        test_point = np.array(test_point).flatten()

        grad, gradient_sec = self.grad_pre_computations(test_point, order)

        if (self.x_trans and self.y_trans):

            mu = self.prediction(test_point_, return_var=False)

            mu = mu.reshape(1,)

            # this is due to the log_10 transformation - check derivation again
            first_der = 10**mu * np.log(10) * grad

            if order == 1:
                return first_der

            else:
                # reshape the gradient vector
                grad_vec = np.atleast_2d(grad)

                term1 = np.log(10) * np.dot(grad_vec.T, grad_vec) + gradient_sec

                second_der = 10**mu * np.log(10) * term1

                return first_der, second_der

        else:
            if order == 1:
                return grad
            else:
                return grad, gradient_sec

    def delete_kernel(self) -> None:
        '''
        Deletes the kernel matrix from the GP module
        '''
        del self.chol_stored
