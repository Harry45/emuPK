# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Calculate the matter power spectrum using CLASS
'''

import numpy as np
from classy import Class


# load the setup file
import settings as st
import cosmology.cosmofuncs as cf
import utils.common as uc


class powerclass(object):
    '''
    Uses CLASS to compute the matter power spectrum
    '''

    def __init__(self):

        # maximum redshift
        self.zmax = st.zmax

        # minimum redshift
        self.zmin = st.zmin

        # minimum k
        self.kmin = st.k_min_h_by_Mpc

        # maximum k
        self.kmax = st.kmax

    def configurations(self) -> None:
        '''
        Calculate and store the basic configurations which will be used by CLASS. This requires
        setting up a dictionary for the quantities we want CLASS to take as default and also the
        quantity we want CLASS to output.
        '''

        # generate base dictionary for CLASS

        if st.mode == 'halofit':
            self.class_args = {'z_max_pk': self.zmax,
                               'output': 'mPk',
                               'non linear': st.mode,
                               'P_k_max_h/Mpc': st.k_max_h_by_Mpc,
                               'halofit_k_per_decade': st.halofit_k_per_decade,
                               'halofit_sigma_precision': st.halofit_sigma_precision}

        # need to check addtional inputs if we use HMcode
        else:
            self.class_args = {'z_max_pk': self.zmax,
                               'output': 'mPk',
                               'non linear': st.mode,
                               'P_k_max_h/Mpc': st.k_max_h_by_Mpc,
                               'eta_0': st.eta,
                               'cmin': st.cmin}

        # redshift range
        self.redshifts = np.linspace(self.zmin, self.zmax, st.nz, endpoint=True)

        # k range
        self.k_range = np.geomspace(self.kmin, self.kmax, st.nk, endpoint=True)

    def pk_nonlinear(self, parameters: dict, **kwargs) -> np.ndarray:
        '''
        Calculate the 3D matter power spectrum based on the emulator setting file

        :param: parameters (dict) - inputs to calculate the matter power spectrum

        :return: pk_matter (np.ndarray) - the 3D matter power spectrum
        '''

        # Calculate the 3D matter power spectrum
        class_module = self.class_compute(parameters)

        # Get the Hubble parameter
        h = class_module.h()

        if 'z' in kwargs:
            z0 = kwargs.pop('z')

            pk_matter = np.zeros(st.nk, 'float64')

            for k in range(st.nk):
                pk_matter[k] = class_module.pk(self.k_range[k] * h, z0)

        else:

            # Get power spectrum P(k=l/r,z(r)) from cosmological module
            pk_matter = np.zeros((st.nk, st.nz), 'float64')

            for k in range(st.nk):
                for z in range(st.nz):

                    # get the matter power spectrum
                    pk_matter[k, z] = class_module.pk(self.k_range[k] * h, self.redshifts[z])

        # clean class_module to prevent memory issue
        cf.delete_module(class_module)

        return pk_matter

    def pk_nonlinear_timed(self, parameters: dict) -> dict:
        '''
        Calculate the non linear matter power spectrum but within an allocated period of time

        :param: parameters (dict) - a dictionary for inputs to CLASS

        :return: quant (dict) - a dictionary of the calculated quantities
        '''

        if st.components:
            state, quant = cf.runTime(cf.timeOut, self.pk_nonlinear_components, parameters, st.timeout)
        else:
            state, quant = cf.runTime(cf.timeOut, self.pk_nonlinear, parameters, st.timeout)

        # add perturbation until run is successful
        while state is False:

            print('CLASS did not run successfully - adding small perturbation to parameter vector.')

            # get the cosmology values
            cosmologies = uc.dvalues(parameters)

            # add a small perturbation to it
            cosmologies += 1.E-4 * np.random.randn(len(parameters))

            # create dictionary
            parameters = cf.mk_dict(st.cosmology, cosmologies)

            if st.components:
                state, quant = cf.runTime(cf.timeOut, self.pk_nonlinear_components, parameters, st.timeout)
            else:
                state, quant = cf.runTime(cf.timeOut, self.pk_nonlinear, parameters, st.timeout)

        return quant

    def pk_nonlinear_components(self, parameters: dict, zref: float = 0.0, **kwargs) -> dict:
        '''
        The non-linear 3D matter power spectrum can be decomposed into:

        P_nonlinear (k,z) = A(z) [1 + q(k,z)] P_linear (k,z0)

        Calculates the following quantities:

        - non linear 3D matter power spectrum, P_nonlinear(k,z)
        - linear matter power spectrum (at the reference redshift - see setting file)
        - the growth factor, A(z)
        - the quantity q(k,z)

        :param: parameters (dict) - inputs to calculate the matter power spectrum

        :param: zref (float) - the reference redshift at which the linear matter power spectrum is calculated (DEFAULT: 0.0)

        :return: quantities (dictionary) - dictionary consisting of the nonlinear power spectrum, linear power spectrum, growth factor and the non-linear function q(k,z)
        '''

        # Calculate the 3D matter power spectrum
        class_module = self.class_compute(parameters)

        # Get the Hubble parameter
        h = class_module.h()

        quantities = {}

        # calculate the linear matter spectrum at reference redshift
        pk_linear = np.array([class_module.pk_lin(self.k_range[k] * h, zref) for k in range(st.nk)])

        # if we specify a redshift, then the power spectrum is calculated at this redshift only
        if 'z' in kwargs:

            z = kwargs.pop('z')

            # calculate the non-linear matter power spectrum
            pk_matter = np.array([class_module.pk(self.k_range[k] * h, z) for k in range(st.nk)])

            # calculate growth factor at fixed redshift
            growth_factor = class_module.pk_lin(self.k_range[0] * h, z) / pk_linear[0]

            # calculate the q function
            q_function = pk_matter / (growth_factor * pk_linear) - 1.0

        else:

            # Get power spectrum P(k=l/r,z(r)) from cosmological module
            pk_matter = np.zeros((st.nk, st.nz), 'float64')

            # empty array for the grwoth factor
            growth_factor = np.zeros(st.nz)

            for k in range(st.nk):
                for z in range(st.nz):

                    # get the matter power spectrum
                    pk_matter[k, z] = class_module.pk(self.k_range[k] * h, self.redshifts[z])

            # calculate the growth factor
            for z in range(st.nz):

                # calculate growth factor at fixed redshift
                growth_factor[z] = class_module.pk_lin(self.k_range[0] * h, self.redshifts[z]) / pk_linear[0]

            # calculate the q function
            q_function = pk_matter / (np.dot(pk_linear.reshape(st.nk, 1), growth_factor.reshape(1, st.nz))) - 1.0

        # append all calculated quantities in a dictionary
        # non-linear matter power spectrum
        quantities['pk'] = pk_matter

        # linear matter power spectrum
        quantities['pl'] = pk_linear

        # the q function
        quantities['qf'] = q_function

        # the growth factor
        quantities['gf'] = growth_factor

        # clean class_module to prevent memory issue
        cf.delete_module(class_module)

        return quantities

    def class_compute(self, parameters: dict):
        '''
        Calculate the relevant quantities using CLASS

        :param: parameters (dict) : dictionary of input parameters to CLASS

        :return: class_module : the whole CLASS module (which contains distances, age, temperature and others)
        '''

        # instantiate Class
        class_module = Class()

        # get the cosmology, nuisance and neutrino parameters
        cosmo, other, neutrino = cf.dictionary_params(parameters)

        # set cosmology
        class_module.set(cosmo)

        # set other configurations (for neutrino)
        class_module.set(other)

        # configuration for neutrino
        class_module.set(neutrino)

        # set basic configurations for Class
        class_module.set(self.class_args)

        # compute the important quantities
        class_module.compute()

        return class_module

    def derivatives(self, params: dict, order: int = 1, eps: list = [1E-3], **kwargs) -> np.ndarray:
        '''
        Calculates the gradient of the power spectrum at a point in parameter space

        :param: params (dict) : a point within the prior box

        :param: eps (float) : epsilon - using central finite difference method to calculate gradient

        :return: grad (list) : an array containing the gradient of each parameter (of size (nk x nz) x ndim), for example 800 x 7
        '''

        params_keys = params.keys()

        params_vals = list(params.values())

        nparams = len(params_vals)

        assert order in [1, 2], 'First and Second derivatives are supported'

        assert len(params_vals) == len(eps), 'Step size mismatch with parameters'

        if order == 1:

            if 'z' in kwargs:

                grad = np.zeros((nparams, st.nk))

            else:

                grad = np.zeros((nparams, st.nk * st.nz))

            for i in range(nparams):

                # make a copy of the parameters
                point_p = np.copy(params_vals)
                point_m = np.copy(params_vals)

                # forward and backward difference
                point_p[i] += eps[i]
                point_m[i] -= eps[i]

                # inputs should be dictionaries
                point_p = cf.mk_dict(params_keys, point_p)
                point_m = cf.mk_dict(params_keys, point_m)

                pk_p = self.pk_nonlinear(point_p, **kwargs)
                pk_m = self.pk_nonlinear(point_m, **kwargs)

                grad_calc = (pk_p - pk_m) / (2 * eps[i])

                grad[i] = grad_calc.flatten()

            return grad.T

        else:

            if 'z' in kwargs:
                grad = np.zeros((nparams, st.nk))

                hessian = np.zeros((nparams, nparams, st.nk))

            else:
                grad = np.zeros((nparams, st.nk * st.nz))

                hessian = np.zeros((nparams, nparams, st.nk * st.nz))

            # calculate the power spectrum at the parameter
            pk = self.pk_nonlinear(params, **kwargs)

            for i in range(nparams):

                # make a copy of the parameters
                point_pi = np.copy(params_vals)
                point_mi = np.copy(params_vals)

                # forward and backward difference
                point_pi[i] += eps[i]
                point_mi[i] -= eps[i]

                point_pi = cf.mk_dict(params_keys, point_pi)
                point_mi = cf.mk_dict(params_keys, point_mi)

                pk_pi = self.pk_nonlinear(point_pi, **kwargs)
                pk_mi = self.pk_nonlinear(point_mi, **kwargs)

                grad_calc = (pk_pi - pk_mi) / (2 * eps[i])

                grad[i, :] = grad_calc.flatten()

                for j in range(nparams):

                    # we need another set for the second derivatives
                    point_pp = np.copy(params_vals)
                    point_pm = np.copy(params_vals)
                    point_mp = np.copy(params_vals)
                    point_mm = np.copy(params_vals)

                    if i == j:

                        hessian_calc = (pk_pi - 2 * pk + pk_mi) / eps[i]**2

                    else:
                        point_pp[i] += eps[i]
                        point_pp[j] += eps[j]

                        point_pm[i] += eps[i]
                        point_pm[j] -= eps[j]

                        point_mp[i] -= eps[i]
                        point_mp[j] += eps[j]

                        point_mm[i] -= eps[i]
                        point_mm[j] -= eps[j]

                        # we need the inputs to be dictionaries
                        point_pp = cf.mk_dict(params_keys, point_pp)
                        point_pm = cf.mk_dict(params_keys, point_pm)
                        point_mp = cf.mk_dict(params_keys, point_mp)
                        point_mm = cf.mk_dict(params_keys, point_mm)

                        # run CLASS at these points
                        pk_pp = self.pk_nonlinear(point_pp, **kwargs)
                        pk_pm = self.pk_nonlinear(point_pm, **kwargs)
                        pk_mp = self.pk_nonlinear(point_mp, **kwargs)
                        pk_mm = self.pk_nonlinear(point_mm, **kwargs)

                        hessian_calc = (pk_pp - pk_pm - pk_mp + pk_mm) / (4 * eps[i] * eps[j])

                    hessian[i, j, :] = hessian_calc.flatten()

            return grad.T, hessian
