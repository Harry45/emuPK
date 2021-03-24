# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Calculate the Weak Lening Power spectra using simulator/emulator
'''


from typing import Tuple
import numpy as np
from classy import Class

# our Python Scripts
import settings as st
import cosmology.cosmofuncs as cf
import cosmology.spectrumcalc as sp
import utils.common as uc
import cosmology.redshift as cr


class spectra(sp.matterspectrum, cr.nz_dist):

    def __init__(self, emu: bool = False, dir_gp: str = 'semigps'):

        # emulator or simulator
        self.emu = emu

        # specify the GP directory
        self.dir_gp = dir_gp

        # set the module to calculate the 3D matter power spectrum
        sp.matterspectrum.__init__(self, emu)

        if self.emu:
            sp.matterspectrum.load_gps(self, dir_gp)

        # set the ell modes
        self.ells_sum = np.linspace(st.ell_min, st.ell_max, st.nell_int)

        # these are the l-nodes for the derivation of the theoretical cl
        self.ells = np.geomspace(st.ell_min, st.ell_max, st.nellsmax)

        # normalisation factor
        self.ell_norm = self.ells_sum * (self.ells_sum + 1) / (2. * np.pi)

    def n_of_z(self, zcenter: list, model_name: str, dist_prop: dict, dist_range: dict = {}) -> dict:
        '''
        Calculate the (mid-) redshift and the heights

        :param: zcenter (list) - a list of the center of source distribution

        :param: nodel_name (str) - name of the n(z) we want to use - the following currently supported

        1) model_1
        2) model_2
        3) gaussian

        :param: dist_range (dict) - a dictionary with the following key words: zmin, zmax, nzmax

        :param: dist_prop (dict) - a dictionary with the key words for the specific distribution,for example:

        1) nz_model_2: dist_prop = {alpha: 2, beta: 1.5}
        2) nz_gaussian: dist_prop = {sigma: [0.25, 0.25]}

        if we are using 2 tomographic bins in the latter

        :return: red_args (dict) - a dictionary with the redshift and heights
        '''

        # set the module for the redshift distribution
        cr.nz_dist.__init__(self, **dist_range)

        model_func = getattr(cr.nz_dist, 'nz_' + model_name)

        red_args = {}

        self.nzbins = len(zcenter)

        for i in range(self.nzbins):
            if model_name == 'model_1':
                # we just need to bin centre
                red_args['h' + str(i)] = model_func(self, zcenter[i])

            elif model_name == 'gaussian':
                # for Gaussian distribution, we require individual standard deviation
                sigma = dist_prop['sigma'][i]
                red_args['h' + str(i)] = model_func(self, zcenter[i], sigma)

            else:
                # here we just need alpha and beta
                red_args['h' + str(i)] = model_func(self, zcenter[i], **dist_prop)

        # record redshift in the dictionary
        red_args['z'] = self.mid_z

        # we need it to do other calculations
        self.zh = red_args

        # we also need the number of redshifts to calculate the kernels
        self.nzmax = len(self.mid_z)

        return red_args

    def pk_matter(self, cosmo: dict, a_bary: float = 0.0) -> Tuple[dict, dict]:
        '''
        Calculate the non-linear matter power spectrum

        :param: d (dict) - a dictionary with all the parameters (keys and values)

        :return: pk_matter (np.ndarray), quant (dict) - an array for the non-linear matter power spectrum
        and a dictionary with the important quantities related to cosmology
        '''

        # extract redshift
        redshifts = self.zh['z']

        # calculate the basic quantities
        quant = self.basic_class(cosmo)

        # comoving radial distance
        chi = quant['chi']

        # we use the values of k and z where the GPs are built to build the interpolator
        k = self.k_range
        z = self.redshifts

        # get the predicted quantities from emulator or simulator
        if st.components:
            gf, spectrum, pl = sp.matterspectrum.int_pk_nl(self, params=cosmo, a_bary=a_bary, k=k, z=z)

        else:
            spectrum = sp.matterspectrum.int_pk_nl(self, params=cosmo, a_bary=a_bary, k=k, z=z)

        # emulator is trained with k in units of h Mpc^-1
        # therefore, we should input k = k/h in interpolator
        # example: interp(*[np.log(0.002/d['h']), 2.0])
        inputs = [k, z, spectrum.flatten()]

        interp = uc.like_interp_2d(inputs)

        # Get power spectrum P(k=l/r,z(r)) from cosmological module or emulator
        pk_matter = np.zeros((st.nellsmax, chi.shape[0]), 'float64')
        k_max_in_inv_mpc = st.kmax * cosmo['h']

        for il in range(st.nellsmax):
            for iz in range(1, chi.shape[0]):

                k_in_inv_mpc = (self.ells[il] + 0.5) / chi[iz]

                if k_in_inv_mpc > k_max_in_inv_mpc:

                    # assign a very small value of matter power
                    pk_dm = 1E-300

                else:

                    # the interpolator is built on top of log(k[h/Mpc])
                    newpoint = [np.log(k_in_inv_mpc / cosmo['h']), redshifts[iz]]

                    # predict the power spectrum
                    pk_dm = interp(*newpoint)

                # record the matter power spectrum
                pk_matter[il, iz] = pk_dm

        # record A_factor in the quant dictionary
        quant['a_fact'] = (3. / 2.) * quant['omega_m'] * quant['small_h']**2 / 2997.92458**2

        return pk_matter, quant

    def wl_power_spec(self, cosmo: dict, a_bary: float = 0.0) -> Tuple[dict, dict, dict]:
        '''
        Power spectrum calculation using the functional form of the n(z) distribution

        :param: d (dict) - a dictionary for the parameters
        '''

        # get matter power spectrum and important quantities
        pk_matter, quant = self.pk_matter(cosmo, a_bary)

        # get the comoing radial distance
        chi = quant['chi']

        # A factor
        a_fact = quant['a_fact']

        # get the n(z) distributions
        zh = self.zh

        # n(z) to n(chi)
        pr_chi = np.array([zh['h' + str(i)] * quant['dzdr'] for i in range(self.nzbins)]).T

        kernel = np.zeros((self.nzmax, self.nzbins), 'float64')

        for zbin in range(self.nzbins):
            for iz in range(1, self.nzmax):
                fun = pr_chi[iz:, zbin] * (chi[iz:] - chi[iz]) / chi[iz:]
                kernel[iz, zbin] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (chi[iz + 1:] - chi[iz:-1]))
                kernel[iz, zbin] *= chi[iz] * (1. + self.mid_z[iz])

        # Start loop over l for computation of C_l^shear
        cl_gg_int = np.zeros((self.nzmax, self.nzbins, self.nzbins), 'float64')
        cl_ii_int = np.zeros_like(cl_gg_int)
        cl_gi_int = np.zeros_like(cl_gg_int)

        ps_ee = np.zeros((st.nellsmax, self.nzbins, self.nzbins), 'float64')
        ps_ii = np.zeros_like(ps_ee)
        ps_gi = np.zeros_like(ps_ee)

        # difference in chi (delta chi)
        dchi = chi[1:] - chi[:-1]

        # il refers to index ell
        for il in range(st.nellsmax):

            # find cl_int = (g(r) / r)**2 * P(l/r,z(r))
            for z1 in range(self.nzbins):
                for z2 in range(z1 + 1):

                    factor_ia = cf.get_factor_ia(quant, self.mid_z, 1.0)[1:]
                    fact_ii = pr_chi[1:, z1] * pr_chi[1:, z2] * factor_ia**2 / chi[1:]**2
                    fact_gi = kernel[1:, z1] * pr_chi[1:, z2] + kernel[1:, z2] * pr_chi[1:, z1]
                    fact_gi *= factor_ia / chi[1:]**2

                    cl_gg_int[1:, z1, z2] = kernel[1:, z1] * kernel[1:, z2] / chi[1:]**2 * pk_matter[il, 1:]
                    cl_ii_int[1:, z1, z2] = fact_ii * pk_matter[il, 1:]
                    cl_gi_int[1:, z1, z2] = fact_gi * pk_matter[il, 1:]

            for z1 in range(self.nzbins):
                for z2 in range(z1 + 1):
                    ps_ee[il, z1, z2] = np.sum(0.5 * (cl_gg_int[1:, z1, z2] + cl_gg_int[:-1, z1, z2]) * dchi)
                    ps_ii[il, z1, z2] = np.sum(0.5 * (cl_ii_int[1:, z1, z2] + cl_ii_int[:-1, z1, z2]) * dchi)
                    ps_gi[il, z1, z2] = np.sum(0.5 * (cl_gi_int[1:, z1, z2] + cl_gi_int[:-1, z1, z2]) * dchi)

                    ps_ee[il, z1, z2] *= a_fact**2
                    ps_gi[il, z1, z2] *= a_fact

        cl_ee = {}
        cl_gi = {}
        cl_ii = {}

        for z1 in range(self.nzbins):
            for z2 in range(z1 + 1):
                idx = str(z1) + str(z2)
                cl_ee[idx] = self.ell_norm * uc.interpolate([self.ells, ps_ee[:, z1, z2], self.ells_sum])
                cl_gi[idx] = self.ell_norm * uc.interpolate([self.ells, ps_gi[:, z1, z2], self.ells_sum])
                cl_ii[idx] = self.ell_norm * uc.interpolate([self.ells, ps_ii[:, z1, z2], self.ells_sum])

        return cl_ee, cl_gi, cl_ii

    def basic_class(self, cosmology: dict) -> dict:
        '''
        Calculates basic quantities using CLASS

        :param: d (dict) - a dictionary containing the cosmological and nuisance parameters

        :return: quant (dict) - a dictionary with the basic quantities
        '''

        cosmo, other, neutrino = cf.dictionary_params(cosmology)

        module = Class()

        # input cosmologies
        module.set(cosmo)

        # other settings for neutrino
        module.set(other)

        # neutrino settings
        module.set(neutrino)

        # compute basic quantities
        module.compute()

        # Omega_matter
        omega_m = module.Omega_m()

        # h parameter
        small_h = module.h()

        # critical density
        rc = cf.get_critical_density(small_h)

        # derive the linear growth factor D(z)
        lgr = np.zeros_like(self.mid_z)

        for iz, red in enumerate(self.mid_z):

            # compute linear growth rate
            lgr[iz] = module.scale_independent_growth_factor(red)

            # normalise linear growth rate at redshift = 0
            lgr /= module.scale_independent_growth_factor(0.)

        # get distances from cosmo-module
        chi, dzdr = module.z_of_r(self.mid_z)

        # numerical stability for chi
        chi += 1E-10

        # delete CLASS module to prevent memory overflow
        cf.delete_module(module)

        quant = {'omega_m': omega_m, 'small_h': small_h, 'chi': chi, 'dzdr': dzdr, 'lgr': lgr, 'rc': rc}

        # record the redshift as well

        quant['z'] = self.mid_z

        return quant
