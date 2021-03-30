# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Module to generate the matter power spectrum either from CLASS or using the emulator
'''

import numpy as np

# our scripts
import cosmology.spectrumclass as cl
import cosmology.cosmofuncs as cf
import utils.helpers as hp
import utils.gp as ug
import utils.common as uc
import settings as st


class matterspectrum(cl.powerclass):
    '''
    Routine to sample the cosmological and nuisance parameters. We have variaous options here.
    We can either use the emulator to calculate the 3D matter power spectrum or we can use CLASS itself.
    '''

    def __init__(self, emulator=True):
        '''
        :param: emulator (bool) - if True, the emulator is used, else CLASS is used
        '''

        self.emulator = emulator

        # new redshift range
        self.z_new = np.linspace(st.zmin, st.zmax, st.nz_new, endpoint=True)

        # new k range
        self.k_new = np.geomspace(st.k_min_h_by_Mpc, st.kmax, st.nk_new, endpoint=True)

        # Call some basic quantities/functions from CLASS script
        cl.powerclass.__init__(self)
        cl.powerclass.configurations(self)

    def load_gps(self, directory: str = 'semigps') -> list:
        '''
        Load all the trained Gaussian Processes.

        :param: directory (str) - the directory where the GPs are stored (depends on our choice:

        - zero mean GP
        - semi-GP (DEFAULT: semigps)

        :return: gps (list) - a list containing all the GPs
        '''

        if st.components and not st.neutrino:
            # folders where we want to store the GPs
            folder_gf = directory + '/pknl_components' + st.d_one_plus + '/gf'
            folder_qf = directory + '/pknl_components' + st.d_one_plus + '/qf'
            folder_pl = directory + '/pknl_components' + st.d_one_plus + '/pl'

            self.gps_gf = [hp.load_pkl_file(folder_gf, 'gp_' + str(i)) for i in range(st.nz)]
            self.gps_qf = [hp.load_pkl_file(folder_qf, 'gp_' + str(i)) for i in range(st.nz * st.nk)]
            self.gps_pl = [hp.load_pkl_file(folder_pl, 'gp_' + str(i)) for i in range(st.nk)]

        if st.components and st.neutrino:
            raise ValueError('Neutrino - not yet implemented')

        if not st.components and not st.neutrino:

            folder = directory + '/pknl' + st.d_one_plus

            self.gps_pknl = [hp.load_pkl_file(folder, 'gp_' + str(i)) for i in range(st.nz * st.nk)]

        if not st.components and st.neutrino:
            raise ValueError('Neutrino - not yet implemented')

    def mean_prediction(self, parameters: np.ndarray) -> np.ndarray:
        '''
        Calculate the mean prediction from all the GPs on the grid k x z

        :param: parameters (np.ndarray) - a vector of the test point in parameter space

        :return: pred (np.ndarray) - the predictions from all the GPs
        '''

        if st.components:

            # generate arguments for each component
            args_gf = list(zip([parameters] * st.nz, self.gps_gf))
            args_qf = list(zip([parameters] * st.nz * st.nk, self.gps_qf))
            args_pl = list(zip([parameters] * st.nk, self.gps_pl))

            # prediction based on whether we have chosen to do the y-transformation

            # growth function
            if st.gf_args['y_trans']:

                pred_gf = np.array(list(map(ug.prediction, args_gf)))

            else:

                pred_gf = np.array(list(map(ug.pred_normal, args_gf)))

            # q function
            if st.qf_args['y_trans']:

                pred_qf = np.array(list(map(ug.prediction, args_qf)))

            else:

                pred_qf = np.array(list(map(ug.pred_normal, args_qf)))

            # linear matter power spectrum
            if st.pl_args['y_trans']:

                pred_pl = np.array(list(map(ug.prediction, args_pl)))

            else:

                pred_pl = np.array(list(map(ug.pred_normal, args_pl)))

            # return results in a dictionary
            pred = {'gf': pred_gf, 'qf': pred_qf, 'pl': pred_pl}

            return pred

        if not st.components:

            # generate arguments for predicting the non-linear matter power spectrum
            args_pknl = list(zip([parameters] * st.nz * st.nk, self.gps_pknl))

            # check if we have chosen to transform the power spectrum
            if st.pknl_args['y_trans']:

                pred_pknl = np.array(list(map(ug.prediction, args_pknl)))

            else:

                pred_pknl = np.array(list(map(ug.pred_normal, args_pknl)))

            return pred_pknl

    def pk_nl_pred(self, params: dict) -> np.ndarray:
        '''
        Calculate the non linear matter power spectrum at a given test point

        :param: parameters (np.ndarray) - a vector of the test point in parameter space

        :return: pk (np.ndarray) - the non linear matter power spectrum (reshaped in nk x nz)
        '''

        # get the parameter values
        # input to the emulator should be an array
        parameters = uc.dvalues(params)

        # use CLASS if we want (but this might cause CLASS to get stuck at large P_k_max)
        if not self.emulator and st.components:

            if st.timed:
                rec = cl.powerclass.pk_nonlinear_timed(self, params)
            else:
                rec = cl.powerclass.pk_nonlinear_components(self, params)

            return rec

        elif not self.emulator and not st.components:

            if st.timed:
                pk = cl.powerclass.pk_nonlinear_timed(self, params)
            else:
                pk = cl.powerclass.pk_nonlinear(self, params)

            return pk

        # use the emulator
        elif self.emulator and not st.components:

            # get the non-linear power spectrum directly
            pk = self.mean_prediction(parameters).reshape(st.nk, st.nz)

            return pk

        else:

            # get the 3 components of the non-linear power spectrum
            comp = self.mean_prediction(parameters)

            # product of the growth factor and the linear matter power spectrum
            gf_pl = np.dot(comp['pl'].reshape(st.nk, 1), comp['gf'].reshape(1, st.nz))

            if st.emu_one_plus_q:
                # we emulate 1 + q(k,z)
                pk = comp['qf'].reshape(st.nk, st.nz) * gf_pl

            else:
                # we emulate q(k,z)
                pk = (comp['qf'].reshape(st.nk, st.nz) + 1.0) * gf_pl

            # we return the growth factor, the non-linear matter power spectrum and the linear matter spectrum
            rec = {'pk': pk, 'gf': comp['gf'], 'pl': comp['pl']}

            return rec

    def int_pk_nl(self, params: dict, a_bary: float = 0, int_type: str = 'cubic', **kwargs) -> np.ndarray:
        '''
        Calculate the (interpolated) 3D matter power spectrum at a test point

        :param: parameters (dict) : a dictionary of the test point in parameter space

        :param: a_bary (float) : the baryon feedback parameter (DEFAULT = 0)

        :param: int_type (str) : type of interpolation (linear, cubic, quintic). Default is cubic

        :return: spectrum (np.ndarray) : the interpolated 3D matter power spectrum
        '''

        # calculate the mean prediction from the GPs or CLASS first
        rec = self.pk_nl_pred(params)

        if 'z' in kwargs and 'k' in kwargs:

            # get the wavenumber
            k = kwargs.pop('k')

            # get the redshift
            z = kwargs.pop('z')

        else:

            # stick with the one provided in the setting file
            k = self.k_new

            # and the redshift on a fine grid
            z = self.z_new

        # baryon feedback
        bf = cf.bar_fed(k / params['h'], z, a_bary)

        # grid from the given k and z
        grid = [k, z]

        if (not self.emulator and st.components) or (self.emulator and st.components):

            inputs = [self.k_range, self.redshifts, rec['pk'].flatten(), int_type]

            # the interpolated power spectrum
            spectrum = uc.two_dims_interpolate(inputs, grid).T

            spectrum = bf * spectrum

            # we can also interpolate the other quantities
            # growth factor
            inputs_gf = [self.redshifts, rec['gf'], z]

            int_gf = uc.interpolate(inputs_gf)

            # linear matter power spectrum

            inputs_pl = [self.k_range, rec['pl'], k]

            int_pl = uc.interpolate(inputs_pl)

            return int_gf, spectrum, int_pl

        else:

            inputs = [self.k_range, self.redshifts, rec.flatten(), int_type]

            # the interpolated power spectrum
            spectrum = uc.two_dims_interpolate(inputs, grid).T

            spectrum = bf * spectrum

            return spectrum

    def int_grad_pk_nl(
            self,
            params: dict,
            order: int = 1,
            int_type: str = 'cubic',
            eps: list = [1E-3],
            **kwargs) -> dict:
        '''
        Calculates the gradient of the power spectrum at a point in parameter space

        :param: params (dict) : a point within the prior box

        :param: eps (float) : epsilon - using central finite difference method to calculate gradient

        :param: int_type (str) : type of interpolation (linear, cubic, quintic). Default is cubic

        :return: grad (dict) : a dictionary containing the gradient of each parameter
        '''

        # NOTE : Need to account for interpolation in the case of second derivatives
        # Different format for CLASS (n x n x 800)and GP (to check)

        # get the parameter values
        parameters = uc.dvalues(params)

        # the new grid
        grid = [self.k_new, self.z_new]

        # number of parameters
        nparams = len(parameters)

        if order == 1:

            if self.emulator:

                gradient = self.gp_gradient(parameters, order)

            else:
                gradient = cl.powerclass.derivatives(self, params, order, eps)

            # create an empty dictionary to record interpolated gradient
            grad = {}

            for p in range(nparams):

                # inputs to the interpolator
                inputs = [self.k_range, self.redshifts, gradient[:, p], int_type]

                # the interpolated power spectrum
                grad[str(p)] = uc.two_dims_interpolate(inputs, grid)

            return grad

        else:

            if self.emulator:

                gradient, hessian = self.gp_gradient(parameters, order)

            else:

                gradient, hessian = cl.powerclass.derivatives(self, params, order, eps)

            grad = {}
            hess = {}

            for p in range(nparams):

                # inputs to the interpolator
                inputs = [self.k_range, self.redshifts, gradient[:, p], int_type]

                # the interpolated power spectrum
                grad[str(p)] = uc.two_dims_interpolate(inputs, grid)

                for q in range(nparams):

                    inputs = [self.k_range, self.redshifts, hessian[p, q, :], int_type]

                    hess[str(p) + str(q)] = uc.two_dims_interpolate(inputs, grid)

            return grad, hess

    def gp_gradient(self, testpoint: np.ndarray, order: int = 1) -> np.ndarray:
        '''
        Calculate the gradient of the power spectrum

        :param: testpoint (np.ndarray) : a testpoint

        :param: order (int) : 1 or 2 (1 refers to first derivative and 2 refers to second)

        :return: the derivatives of the power spectrum (of size (nk x nz) x ndim), for example 800 x 7
        '''

        # number of parameters
        nparams = len(testpoint)

        # number of GPs if we are emulating power spectrum directly
        ngps = st.nk * st.nz

        if not st.components and order == 1:

            arguments = list(zip([testpoint] * ngps, self.gps_pknl, [order] * ngps))

            results = np.array(list(map(ug.gradient, arguments)))

        elif not st.components and order == 2:

            arguments = list(zip([testpoint] * ngps, self.gps_pknl, [order] * ngps))

            dummy = list(map(ug.gradient, arguments))

            # the gradient of shape: ngps x nparams
            grad = np.array([dummy[i][0] for i in range(ngps)])

            # the hessian of shape: nparams x nparams x ngps
            hess = np.array([dummy[i][1] for i in range(ngps)])

            results = (grad, hess.T)

        elif st.components and order == 1:

            # growth factor
            args_gf = list(zip([testpoint] * st.nz, self.gps_gf, [order] * st.nz))

            # q function
            args_qf = list(zip([testpoint] * st.nk * st.nz, self.gps_qf, [order] * st.nk * st.nz))

            # linear power spectrum
            args_pl = list(zip([testpoint] * st.nk, self.gps_pl, [order] * st.nk))

            # calculate the gradients with the GPs
            grad_gf = np.array(list(map(ug.gradient, args_gf)))
            grad_qf = np.array(list(map(ug.gradient, args_qf)))
            grad_pl = np.array(list(map(ug.gradient, args_pl)))

            # calculate the gradient (chain rule)
            # we need the mean prediction from the GPs
            rec = self.mean_prediction(testpoint)

            # reshaped arrays for gradient calculation
            g_gf_r = grad_gf.reshape(1, st.nz, nparams)
            g_qf_r = grad_qf.reshape(st.nk, st.nz, nparams)
            g_pl_r = grad_pl.reshape(st.nk, 1, nparams)

            # reshape the mean prediction
            gf_r = rec['gf'].reshape(1, st.nz, 1)
            qf_r = rec['qf'].reshape(st.nk, st.nz, 1)
            pl_r = rec['pl'].reshape(st.nk, 1, 1)

            # calculate the 3 terms for the gradients
            if st.emu_one_plus_q:
                t1 = g_gf_r * qf_r * pl_r
                t3 = gf_r * qf_r * g_pl_r
            else:
                t1 = g_gf_r * (1. + qf_r) * pl_r
                t3 = gf_r * (1. + qf_r) * g_pl_r

            t2 = gf_r * g_qf_r * pl_r

            # gradient is sum of the 3 terms
            results = t1 + t2 + t3

            # reshape the gradient
            results = results.reshape(st.nk * st.nz, nparams)

        elif st.components and order == 2:

            # growth factor
            args_gf = list(zip([testpoint] * st.nz, self.gps_gf, [order] * st.nz))

            # q function
            args_qf = list(zip([testpoint] * st.nk * st.nz, self.gps_qf, [order] * st.nk * st.nz))

            # linear power spectrum
            args_pl = list(zip([testpoint] * st.nk, self.gps_pl, [order] * st.nk))

            # get the first and second derivatives for
            # growth factor
            # q function
            # linear matter power spectrum
            der_gf = list(map(ug.gradient, args_gf))
            der_qf = list(map(ug.gradient, args_qf))
            der_pl = list(map(ug.gradient, args_pl))

            # -------------------------------------------------------------------------------------
            # calculate the gradient (chain rule)
            grad_gf = np.array([der_gf[i][0] for i in range(st.nz)])
            grad_qf = np.array([der_qf[i][0] for i in range(ngps)])
            grad_pl = np.array([der_pl[i][0] for i in range(st.nk)])

            # we need the mean prediction from the GPs
            rec = self.mean_prediction(testpoint)

            # reshape arrays for gradient calculation
            g_gf_r = grad_gf.reshape(1, st.nz, nparams)
            g_qf_r = grad_qf.reshape(st.nk, st.nz, nparams)
            g_pl_r = grad_pl.reshape(st.nk, 1, nparams)

            # reshape the mean prediction
            gf_r = rec['gf'].reshape(1, st.nz, 1)
            qf_r = rec['qf'].reshape(st.nk, st.nz, 1)
            pl_r = rec['pl'].reshape(st.nk, 1, 1)

            # calculate the 3 terms for the gradients
            if st.emu_one_plus_q:
                t1 = g_gf_r * qf_r * pl_r
                t3 = gf_r * qf_r * g_pl_r
            else:
                t1 = g_gf_r * (1. + qf_r) * pl_r
                t3 = gf_r * (1. + qf_r) * g_pl_r

            t2 = gf_r * g_qf_r * pl_r

            # gradient is sum of the 3 terms
            grad = t1 + t2 + t3

            # reshape the gradient
            grad = grad.reshape(st.nk * st.nz, nparams)

            # -------------------------------------------------------------------------------------
            # calculate the hessian (chain rule)
            hess_gf = np.array([der_gf[i][1] for i in range(st.nz)])
            hess_qf = np.array([der_qf[i][1] for i in range(ngps)])
            hess_pl = np.array([der_pl[i][1] for i in range(st.nk)])

            # reshape arrays for second derivatives calculation
            h_gf_r = hess_gf.reshape(1, st.nz, nparams, nparams)
            h_qf_r = hess_qf.reshape(st.nk, st.nz, nparams, nparams)
            h_pl_r = hess_pl.reshape(st.nk, 1, nparams, nparams)

            # reshape arrays for the mean prediction
            gf_r_ = rec['gf'].reshape(1, st.nz, 1, 1)
            qf_r_ = rec['qf'].reshape(st.nk, st.nz, 1, 1)
            pl_r_ = rec['pl'].reshape(st.nk, 1, 1, 1)

            # reshape arrays for first derivatives
            g_gf_r_ = grad_gf.reshape(1, st.nz, nparams, 1)
            g_qf_r_ = grad_qf.reshape(st.nk, st.nz, 1, nparams)
            g_pl_r_ = grad_pl.reshape(st.nk, 1, 1, nparams)

            # for the hessian parts
            if st.emu_one_plus_q:
                t1 = h_gf_r * qf_r_ * pl_r_
                t3 = gf_r_ * qf_r_ * h_pl_r
            else:
                t1 = h_gf_r * (1.0 + qf_r_) * pl_r_
                t3 = gf_r_ * (1.0 + qf_r_) * h_pl_r

            t2 = gf_r_ * h_qf_r * pl_r_
            # need to make sure we get nparams x nparams for every product
            t4 = 2.0 * g_gf_r_ * g_qf_r_ * pl_r_

            if st.emu_one_plus_q:
                t5 = 2.0 * g_gf_r_ * qf_r_ * g_pl_r_
            else:
                t5 = 2.0 * g_gf_r_ * (1.0 + qf_r_) * g_pl_r_

            t6 = 2.0 * gf_r_ * grad_qf.reshape(st.nk, st.nz, nparams, 1) * g_pl_r_

            # hessian
            hess = t1 + t2 + t3 + t4 + t5 + t6

            hess = hess.reshape(ngps, nparams, nparams)

            results = (grad, hess.T)

        return results
