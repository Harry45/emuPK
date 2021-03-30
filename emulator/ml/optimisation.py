# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Function to train a GP
'''

import numpy as np

# our scripts
import ml.semigp as sgp
import ml.gaussianlinear as gl
import settings as st


def maximise(x_train: np.ndarray, y_train: np.ndarray, y_trans: bool = True, lambda_cap: float = 1) -> object:
    '''
    Function for training one GP

    :param: x_train (np.ndarray) : the inputs to the GP

    :param: y_train (np.ndarray) : the output from the traning point

    :param: y_trans (bool) : option to transform the target

    :param: lambda_cap (float) : the prior width on the regression coefficients

    :return: gp_module (class) : Python class with the trained GP
    '''

    # ------------------------------------------------------------------------
    # The GLM here is not important - it is just used as a guige to understand the emulating scheme
    # We will instead fit both the regression coefficients and the residulas using the GP
    # instantiate the GLM module
    glm_module = gl.GLM(
        theta=x_train,
        y=y_train,
        order=st.order,
        var=st.var,
        x_trans=st.x_trans,
        y_trans=y_trans,
        use_mean=st.use_mean)

    # make the appropriate transformation
    # rotation of the input parameters
    glm_module.do_transformation()

    # compute the basis functions
    phi = glm_module.compute_basis()

    # set the regression prior
    glm_module.regression_prior(lambda_cap=lambda_cap)

    # compute the log_evidence
    log_evi = glm_module.evidence()

    # calculate the posterior mean and variance of the regression coefficients
    post_beta, cov_beta = glm_module.posterior_coefficients()

    # ------------------------------------------------------------------------

    # number of kernel hyperparameters (amplitude and 7 lengthscales)
    n_params = int(x_train.shape[1] + 1)

    # instantiate the GP module
    gp_module = sgp.GP(theta=x_train, y=y_train, var=st.var, order=st.order, x_trans=st.x_trans, y_trans=y_trans,
                       jitter=st.jitter, use_mean=st.use_mean)

    # Make appropriate transformation
    gp_module.do_transformation()

    # Compute design matrix
    phi_gp = gp_module.compute_basis()

    # Input regression prior
    # (default: 0 mean and unit variance: inputs -> mean = None, cov = None, Lambda = 1)
    mean_ = np.zeros(phi_gp.shape[1])  # post_beta.flatten()
    cov_ = np.identity(phi_gp.shape[1])
    gp_module.regression_prior(mean=mean_, cov=cov_, lambda_cap=lambda_cap)

    # number of kernel hyperparameters
    n_params = x_train.shape[1] + 1

    # Set bound (prior for kernel hyperparameters)
    bnd = np.repeat(np.array([[st.l_min, st.l_max]]), n_params, axis=0)

    # amplitude of the residuals
    res = np.atleast_2d(y_train).T - np.dot(glm_module.phi, post_beta)
    res = res.flatten()
    std = np.std(res) + 1E-300
    amp = 2 * np.log(std)

    # we set a different bound for the amplitude
    # but one could use the answer from the Gaussian Linear Model
    # to propose an informative bound
    bnd[0] = np.array([st.a_min, st.a_max])
    # bnd[0] = np.array([amp - 1, amp + 1])

    # run optimisation
    gp_module.fit(
        method=st.method,
        bounds=bnd,
        options={
            'ftol': st.ftol,
            'maxiter': st.maxiter},
        n_restart=st.n_restart)

    if st.del_kernel:
        gp_module.delete_kernel()

    return gp_module
