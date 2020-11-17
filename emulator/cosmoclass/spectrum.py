# Author: (Dr to be) Arrykrishna Mootoovaloo
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
import setemu as st
import cosmoclass.utils as ut
import utils.powerspec as up

class matterspectrum(object):

    '''
    Calculate the 3D matter power spectrum based on the setting file for the emulator

    :param: zmax - the maximum redshift (for KiDS-450 analysis, this is equal to 4.66 roughly).
    '''

    def __init__(self, zmax:float=st.zmax):

        # maximum redshift
        self.zmax = zmax

    def input_configurations(self) -> None:
        '''
        Calculate and store the basic configurations which will be used by CLASS. This requires setting up a dictionary for the quantities we want CLASS to take as default and also the quantity we want CLASS to output.
        '''

        # generate base dictionary for CLASS
        self.class_args = {'z_max_pk': self.zmax,
                           'output': 'mPk',
                           'non linear': st.mode,
                           'P_k_max_h/Mpc': st.k_max_h_by_Mpc}

        # redshift range
        self.redshifts = np.linspace(0.0, self.zmax, st.nz, endpoint=True)

        # k range
        self.k_range = np.logspace(np.log10(st.k_min_h_by_Mpc), np.log10(st.k_max_h_by_Mpc), st.nk, endpoint=True)

        # new redshift range
        self.z_new = np.linspace(0.0, self.zmax, st.nz_new, endpoint=True)

        # new k range
        self.k_new = np.logspace(np.log10(st.k_min_h_by_Mpc), np.log10(st.k_max_h_by_Mpc), st.nk_new, endpoint=True)

    def interpolated_spectrum(self, testpoint:np.ndarray, int_type: str = 'cubic')->np.ndarray:
        '''
        Mean prediction of the GP at a test point in parameter space

        :param: testpoint (np.ndarray) : a testpoint

        :param: int_type (str) : type of interpolation (linear, cubic, quintic). Default is cubic

        :return: interpolation (np.ndarray) : the interpolated 3D matter power spectrum
        '''

        # calculate the mean prediction
        pred = self.compute_ps(testpoint).flatten()

        # inputs to the interpolator
        inputs = [self.k_range, self.redshifts, pred, int_type]

        # the new grid
        grid = [self.k_new, self.z_new]

        # the interpolated power spectrum
        spectra = up.kz_interpolate(inputs, grid)

        return spectra

    def gradient(self, parameters: np.ndarray, eps: float = 1E-5)->np.ndarray:
        '''
        Calculates the gradient of the power spectrum at a point in parameter space

        :param: parameters (np.ndarray) : a point within the prior box

        :param: eps (float) : epsilon - using central finite difference method to calculate gradient

        :return: grad (list) : an array containing the gradient of each parameter (of size (nk x nz) x ndim), for example 800 x 7 
        '''

        grad = []

        for i in range(len(parameters)):

            point_p = np.copy(parameters)
            point_m = np.copy(parameters)

            point_p[i] = parameters[i] + eps
            point_m[i] = parameters[i] - eps

            pk_p = self.compute_ps(point_p)
            pk_m = self.compute_ps(point_m)

            grad_calc = (pk_p - pk_m)/(2*eps)

            grad.append(grad_calc.flatten())

        grad = np.array(grad).T

        return grad

    def interpolated_gradient(self, parameters:np.ndarray, eps: float = 1E-5, int_type: str = 'cubic')->list:
        '''
        Calculates the gradient of the power spectrum at a point in parameter space

        :param: parameters (np.ndarray) : a point within the prior box

        :param: eps (float) : epsilon - using central finite difference method to calculate gradient

        :param: int_type (str) : type of interpolation (linear, cubic, quintic). Default is cubic

        :return: grad (list) : a list containing the gradient of each parameter
        '''

        first_der = self.gradient(parameters, eps)

        # create an empty list to record interpolated gradient 
        grad_1 = []

        for p in range(len(parameters)):

            # inputs to the interpolator
            inputs = [self.k_range, self.redshifts, first_der[:,p], int_type]

            # the new grid
            grid = [self.k_new, self.z_new]

            # the interpolated power spectrum
            grad = up.kz_interpolate(inputs, grid)

            grad_1.append(grad)

        return grad_1

    def compute_ps(self, parameters: np.ndarray) -> np.ndarray:
        '''
        Calculate the 3D matter power spectrum based on the emulator setting file

        :param: parameters (np.ndarray) - inputs to calculate the matter power spectrum

        :return: pk_matter (np.ndarray) - the 3D matter power spectrum
        '''

        # get the cosmology, nuisance and neutrino parameters
        _, _, _, nuisance = ut.dictionary_params(parameters)

        # Calculate the 3D matter power spectrum
        class_module = self.class_compute(parameters)

        # Get the Hubble parameter
        h = class_module.h()

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        pk_matter = np.zeros((st.nk, st.nz), 'float64')

        for k in range(st.nk):
            for z in range(st.nz):

                # get the matter power spectrum
                pk_dm = class_module.pk(self.k_range[k] * h, self.redshifts[z])

                if st.include_feedback:

                    # Include baryon feedback
                    # k is expected in h/Mpc
                    pk_matter[k, z] = pk_dm * self.bar_fed(self.k_range[k], self.redshifts[z], nuisance['A_bary'])

                else:
                    pk_matter[k, z] = pk_dm

        # clean class_module to prevent memory issue
        ut.delete_module(class_module)

        return pk_matter

    def class_compute(self, parameters: np.ndarray):
        '''
        Calculate the relevant quantities using CLASS

        :param: parameters (np.ndarray) : array of input parameters to CLASS

        :return: class_module : the whole CLASS module (which contains distances, age, temperature and others)
        '''

        # get the cosmology, nuisance and neutrino parameters
        cosmo, other, neutrino, _ = ut.dictionary_params(parameters)

        # instantiate Class
        class_module = Class()

        # set cosmology
        class_module.set(cosmo)

        # set basic configurations for Class
        class_module.set(self.class_args)

        # set other configurations (for neutrino)
        class_module.set(other)

        # configuration for neutrino
        class_module.set(neutrino)

        # compute the important quantities
        class_module.compute()

        return class_module

    def compute_k_min(self, parameters: np.ndarray) -> float:
        '''
        Calculate k_min based on the input LHS configurations

        :param: parameters (np.ndarray) : array of input parameters to CLASS

        :return: k_min (float) : minimum value of k for that set of parameters
        '''

        class_module = self.class_compute(parameters)

        # calculate distances
        chi, _ = class_module.z_of_r(self.redshifts)

        # numerical statibility if we have redshift = 0
        chi += 1E-300

        # get k_min which is equal to ell_min divided by chi_max
        k_min = st.ell_min / chi.max()

        # clean class_module to prevent memory issue
        ut.delete_module(class_module)

        return k_min

    def bar_fed(self, k, redshift, a_bary=1.):
        """
        Fitting formula for baryon feedback following equation 10 and Table 2 from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

        :param: k (np.ndarray): the wavevector

        :param: z (np.ndarray): the redshift

        :param: A_bary (float): the free amplitude for baryon feedback

        :return: b^2(k,z): bias squared
        """

        bm = st.baryon_model

        # k is expected in h/Mpc and is divided in log by this unit...
        x_wav = np.log10(k)

        # calculate a
        a_factor = 1. / (1. + redshift)

        # a squared
        a_sqr = a_factor * a_factor

        a_z = st.cst[bm]['A2'] * a_sqr + st.cst[bm]['A1'] * a_factor + st.cst[bm]['A0']
        b_z = st.cst[bm]['B2'] * a_sqr + st.cst[bm]['B1'] * a_factor + st.cst[bm]['B0']
        c_z = st.cst[bm]['C2'] * a_sqr + st.cst[bm]['C1'] * a_factor + st.cst[bm]['C0']
        d_z = st.cst[bm]['D2'] * a_sqr + st.cst[bm]['D1'] * a_factor + st.cst[bm]['D0']
        e_z = st.cst[bm]['E2'] * a_sqr + st.cst[bm]['E1'] * a_factor + st.cst[bm]['E0']

        # original formula:
        # bias_sqr = 1.-A_z*np.exp((B_z-C_z)**3)+D_z*x*np.exp(E_z*x)
        # original formula with a free amplitude A_bary:
        bias_sqr = 1. - a_bary * (a_z * np.exp((b_z * x_wav - c_z)**3) - d_z * x_wav * np.exp(e_z * x_wav))

        return bias_sqr
