# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Some important functions for power spectrum and likelihood calculation
'''

from typing import Tuple
import multiprocessing
import numpy as np
import settings as st
import utils.common as uc


# list of indices for double sum (Generated once)
INDEX_EE, INDEX_GI = uc.indices(st.nzmax)
Lab_1 = INDEX_GI[0]
Lab_2 = INDEX_GI[1]
Lba_1 = INDEX_GI[2]
Lba_2 = INDEX_GI[3]


def ds_ee(qs: list, quant: dict) -> np.ndarray:
    '''
    Calculates the double sum for the EE power spectrum

    :param: qs (list) - a list of the functions Q

    :param: quant (dict) - a dictionary containing all precomputed and basic quantities

    :return: dsum (np.ndarray) - the double sum
    '''
    # we need 72 values for the comoving radial distance
    chi_alpha = quant['chi'][1:]

    # redshifts
    redshift = quant['z']

    # difference between the redshifts
    dred = np.diff(redshift)

    # Expressions for Q
    q_0 = qs[0][:, INDEX_EE]
    q_1 = qs[1][:, INDEX_EE]
    q_2 = qs[2][:, INDEX_EE]

    # outer product for delta redshift
    prod_red = np.atleast_2d(np.multiply.outer(dred, dred).flatten())

    # outer product for the comoving angular distance
    prod_chi = np.atleast_2d(np.multiply.outer(chi_alpha, chi_alpha).flatten())

    # outer sum for the comoving radial distance
    sum_chi = np.atleast_2d(np.add.outer(chi_alpha, chi_alpha).flatten())

    dsum = prod_red * (q_0 - (sum_chi / prod_chi) * q_1 + q_2 / prod_chi)

    return dsum


def ps_ee(index_i: int, index_j: int, heights: dict, dsum: np.ndarray) -> np.ndarray:
    '''
    Calculates the weak lensing power spectrum using the double sum approach

    :param: index_i (int) - the i^th tomographic bin

    :param: index_j (int) - th j^th tomographic bin

    :param: heights (dict) - a dictionary with keys: h0, h1, h2

    :param: dsum (np.ndarray) - part of the double sum

    :return: wl_ee (np.ndarray)- the EE weak lensing power spectrum
    '''

    h_i = heights['h' + str(index_i)][1:]
    h_j = heights['h' + str(index_j)][1:]

    prod_h = np.atleast_2d(np.multiply.outer(h_i, h_j).flatten())

    wl_ee = prod_h * dsum

    wl_ee = np.sum(wl_ee, axis=1)

    return wl_ee


def ds_ii(f_ii: np.ndarray, quant: dict) -> np.ndarray:
    '''
    Calculates the double sum for the II power spectrum. In this case, it is a single summation

    :param: f_ii (np.ndarray) - an array for the function F_II

    :param: quant (dict) - a dictionary containing all precomputed and basic quantities

    :return: dsum (np.ndarray) - the double sum
    '''

    # redshifts
    redshift = quant['z']

    # difference between the redshifts
    dred = np.diff(redshift)

    dsum = dred * f_ii[:, 1:] * quant['dzdr'][1:]

    return dsum


def ps_ii(index_i: int, index_j: int, heights: dict, dsum: np.ndarray) -> np.ndarray:
    '''
    Calculates the weak lensing power spectrum using the double sum approach

    :param: index_i (int) - the i^th tomographic bin

    :param: index_j (int) - th j^th tomographic bin

    :param: heights (dict) - a dictionary with keys: h0, h1, h2

    :param: dsum (np.ndarray) - part of the double sum

    :return: wl_ii (np.ndarray)- the II weak lensing power spectrum
    '''

    h_i = heights['h' + str(index_i)][1:]
    h_j = heights['h' + str(index_j)][1:]

    wl_ii = np.sum(h_i * h_j * dsum, axis=1)

    return wl_ii


def ds_gi(f_gi: np.ndarray, quant: dict) -> np.ndarray:
    '''
    Calculates the double sum for the II power spectrum. In this case, it is a single summation

    :param: f_ii (np.ndarray) - an array for the function F_II

    :param: quant (dict) - a dictionary containing all precomputed and basic quantities

    :return: dsum (np.ndarray) - the double sum
    '''

    # comoving radial distance
    chi = quant['chi']

    # redshifts
    redshift = quant['z']

    # difference in redshift
    dred = np.diff(redshift)

    T1 = dred[Lab_1 - 1] * dred[Lab_2 - 1]
    T2 = dred[Lba_1 - 1] * dred[Lba_2 - 1]

    C1 = (1. - chi[Lab_2] / chi[Lab_1]) * f_gi[:, Lab_2]
    C2 = (1. - chi[Lba_1] / chi[Lba_2]) * f_gi[:, Lba_1]

    D1 = T1 * C1
    D2 = T2 * C2

    return D1, D2


def ps_gi(index_i: int, index_j: int, heights: dict, dsum: np.ndarray) -> np.ndarray:
    '''
    Calculates the weak lensing power spectrum using the double sum approach

    :param: index_i (int) - the i^th tomographic bin

    :param: index_j (int) - th j^th tomographic bin

    :param: heights (dict) - a dictionary with keys: h0, h1, h2

    :param: dsum (np.ndarray) - part of the double sum

    :return: wl_gi (np.ndarray)- the II weak lensing power spectrum
    '''

    h_i = heights['h' + str(index_i)]
    h_j = heights['h' + str(index_j)]

    T1 = h_i[Lab_1] * h_j[Lab_2]
    T2 = h_i[Lba_1] * h_j[Lba_2]

    wl_gi = np.sum(T1 * dsum[0] + T2 * dsum[1], axis=1)

    return wl_gi


def integration_q_ell(f_ell, chi, order):
    '''
    Calculate Q_ell for all possible pairs of chi

    :param: f_ell (np.ndarray) : array for F_ell(chi) - see notes for further details)

    :param: chi (np.ndarray) : array for the comoving radial distance

    :param: order (int) : either 0 or 1 or 2

    :return: q_ell (np.ndarray) - array of size nells x nz (see notes for further details)
    '''

    # calculate the integrand (39 x 73)
    integrand = np.atleast_2d(np.power(chi, order)) * f_ell

    # number of redshifts (comoving radial distance)
    # should be equal to 73
    n_chi = len(chi)

    # number of ells
    n_ells = f_ell.shape[0]

    q_ell = np.zeros((n_ells, n_chi))

    for index_ell in range(n_ells):
        for i in range(1, n_chi):

            q_ell[index_ell, i] = np.trapz(integrand[index_ell, 0:i], chi[0:i])

    return q_ell


def cosmo_params(d: dict) -> dict:
    '''
    Given a dictionary for all the parameters, this function returns a dictionary only for the inputs to the emulator

    :param: d (dict) - a dictionary with all the parameters (keys and values)

    :return: emu_param (dict) - a dictionary with inputs to the emulator
    '''

    # empty dictionary to record the parameters
    param = {}

    for k in st.cosmology:
        param[k] = d[k]

    return param


def nuisance_params(d: dict) -> dict:
    '''
    Given a dictionary for all the parameters, this function returns a dictionary only for the inputs to the emulator

    :param: d (dict) - a dictionary with all the parameters (keys and values)

    :return: emu_param (dict) - a dictionary with inputs to the emulator
    '''

    # empty dictionary to record the parameters
    param = {}

    for k in st.nuisance:
        param[k] = d[k]

    return param


def mk_dict(l1: list, l2: list):
    '''
    Create a dictionary given a list of string and a list of numbers

    :param: l1 (list) - list of string (parameter names)

    :param: l2 (list) - list of values (values of each parameter)

    :return: d (dict) - a dictionary consisting of the keys and values
    '''

    if len(l1) != len(l2):
        raise ValueError('Mis-match between parameter names and values.')

    d = dict(zip(l1, l2))

    return d


def delete_module(module):
    '''
    Delete Class module - accumulates memory unnecessarily

    :param: module (classy.Class) - the Class module
    '''
    module.struct_cleanup()

    module.empty()

    del module


def marg_params(d: dict):
    '''
    Returns different dictionaries to be used at different parts of the likelihood code.

    :param: d (dict) - a dictionary with all the parameters (keys and values)

    :return: different dictionaries (based on conditions in the setting file)
    '''

    # empty dictionary to record cosmological parameters
    cosmo = {}

    # empty dictionary to record nuisance parameters
    nuisance = {}

    for k in st.cosmology:
        if k != 'M_tot':
            cosmo[k] = d[k]

    for k in st.nuisance:
        nuisance[k] = d[k]

    # additional settings if we want to include neutrinos
    # other = {'N_ncdm': 1.0, 'deg_ncdm': 3.0, 'T_ncdm': 0.71611, 'N_ur': 0.00641}

    # using this for KV-450 survey
    other = {'N_eff': 2.0328, 'N_ncdm': 1, 'T_ncdm': 0.71611}

    # additional settings if we want to include neutrinos
    if st.neutrino:

        # neutrino is marginalised over in the sampling engine
        par_neutrino = d['M_tot']

    else:

        # neutrino is fixed to some value
        par_neutrino = st.fixed_nm['M_tot']

    # neutrino = {'m_ncdm': par_neutrino / other['deg_ncdm']}
    neutrino = {'m_ncdm': par_neutrino}

    return cosmo, other, neutrino, nuisance


def dictionary_params(d: dict) -> Tuple[dict, dict, dict]:
    '''
    A dictionary for storing all the parameters

    The parameters are organised in the following order (using CLASS and MontePython notations):

    - omega_cdm
    - omega_b
    - ln10^{10}A_s
    - n_s
    - h
    - M_tot

    :param: par (np.ndarray): parameters for the inference

    :return: cosmo (dict) : a dictionary for the cosmology setup

    :return: other (dict) : a dictionary for the neutrino settings

    :return: neutrino (dict) : a dictionary for the neutrino

    :return: nuisance (dict) : a dictionary which contains the baryon feedback parameter
    '''

    # In addition to arranging the mass of the neutrinos in a neutrino
    # hierarchy, it is possible to sample the total neutrino mass for a case
    # with three massive neutrinos with degenerate mass. Although not a
    # realistic scenario, it is often sufficient to use three degenerate
    # neutrinos, speeding up computations in
    # the Boltzmann solver.

    # This is done via the input parameter M_tot, remembering to specify the
    # cosmo_arguments from before, but this time with only one type of
    # neutrino species, N_ur=0.00641, N_ncdm=1 and T_ncdm=0.71611, and instead
    # specifying the degeneracy of the neutrino species, deg_ncdm=3. The total
    # neutrino mass is then simply divided by the number of massive neutrino
    # species and the resulting particle mass is passed to CLASS.

    # https://www.groundai.com/project/montepython-3-boosted-mcmc-sampler-and-other-features/2

    # Some explanations here: https://github.com/lesgourg/class_public/issues/70

    # Some explanations here: https://monte-python.readthedocs.io/en/latest/_modules/data.html

    # empty dictionary to record cosmological parameters
    cosmo = {}

    for k in st.cosmology:
        if k != 'M_tot':
            cosmo[k] = d[k]

    # settings for neutrino
    other = {'N_ncdm': 1.0, 'deg_ncdm': 3.0, 'T_ncdm': 0.71611, 'N_ur': 0.00641}

    # additional settings if we want to include neutrinos
    if st.neutrino:

        # neutrino is marginalised over in the sampling engine
        par_neutrino = d['M_tot']

    else:

        # neutrino is fixed to some value
        par_neutrino = st.fixed_nm['M_tot']

    neutrino = {'m_ncdm': par_neutrino / other['deg_ncdm']}

    return cosmo, other, neutrino


def bar_fed(k, z, a_bary=0.0):
    """
    Fitting formula for baryon feedback following equation 10 and Table 2 from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

    :param: k (np.ndarray): the wavevector

    :param: z (np.ndarray): the redshift

    :param: A_bary (float): the free amplitude for baryon feedback (Default: 0.0)

    :return: b^2(k,z): bias squared
    """
    k = np.atleast_2d(k).T

    z = np.atleast_2d(z)

    bm = st.baryon_model

    # k is expected in h/Mpc and is divided in log by this unit...
    x_wav = np.log10(k)

    # calculate a
    a_factor = 1. / (1. + z)

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


def get_factor_ia(quant: dict, redshift: np.ndarray, amplitude: float, exponent=0.0) -> np.ndarray:
    '''
    Calculates F(chi) - equation 23 in Kohlinger et al. 2017.

    :param: quant (dict) - a dictionary containingthe critical density, omega matter, linear groth rate, Hubble parameter

    :param: redshift (np.ndarray) - a vector for the redshift

    :param: amplitude (float) - the amplitude due to intrinsic alignment

    :param: exponent (float) - an exponential factor (default: 0.0) - not used in inference
    '''

    # critical density
    rc = quant['rc']

    # omega matter
    om = quant['omega_m']

    # linear growth rate
    lgr = quant['lgr']

    # Hubble parameter
    h = quant['small_h']

    # in Mpc^3 / M_sol
    const = 5E-14 / h**2

    # arbitrary convention
    redshift_0 = 0.3

    # additional term for the denominator (not in paper)
    denom = ((1. + redshift) / (1. + redshift_0))**exponent

    # factor = (-1. * amplitude * const * rc * om) / (lgr * denom)
    factor = (amplitude * const * rc * om) / (lgr * denom)

    return factor


def get_critical_density(small_h):
    """
    The critical density of the Universe at redshift 0.

    :param: small_h (float) - the Hubble parameter

    :return: rho_crit_0 (float) - the critical density at redshift zero
    """

    # Mpc to cm conversion
    mpc_cm = 3.08568025e24

    # Mass of Sun in grams
    mass_sun_g = 1.98892e33

    # Gravitational constant
    grav_const_mpc_mass_sun_s = mass_sun_g * (6.673e-8) / mpc_cm**3.

    # in s^-1 (to check definition of this)
    h_100_s = 100. / (mpc_cm * 1.0e-5)

    rho_crit_0 = 3. * (small_h * h_100_s)**2. / (8. * np.pi * grav_const_mpc_mass_sun_s)

    return rho_crit_0


def timeOut(func: object, param: dict, q: object) -> None:
    '''
    Calculate a (general) function within an allocated time frame

    :param: func (object) - the module for calculating the non-linear matter power spectrum

    :param: param (dict) - a dictionary of input cosmology

    :param: q (object) - the multiprocessing queue
    '''
    quantities = func(param)

    q.put(quantities)


def timeOutComponents(func: object, param: dict, q: object) -> None:
    '''
    Calculate the components of the non-linear matter power spectrum

    :param: func (object) - the module for calculating the non-linear matter power spectrum

    :param: param (dict) - a dictionary of input cosmology

    :param: q (object) - the multiprocessing queue
    '''

    quantities = func.pk_nonlinear_components(param)

    q.put(quantities)


def runTime(target: object, func: object, param: dict, timeout: int = 60):
    '''
    Execute the above function(s) within the allocated time frame

    :param: target (object) - the target we want to time

    :param: func (object) - the module for calculating the non-linear matter power spectrum

    :param: param (dict) - a dictionary of input cosmology

    :param: timeout (int) - the alloated time window to allow the code to run

    :return: state (bool) - True if CLASS runs successfully

    :return: results (dict) - if CLASS runs successfully
    '''

    queue = multiprocessing.Queue()

    # Start bar as a process
    process = multiprocessing.Process(target=target, args=(func, param, queue,))

    # start the process
    process.start()

    # Wait for x seconds or until process finishes
    process.join(timeout)

    queueSize = queue.qsize()

    if queueSize == 0:

        # Terminate - may not work if process is stuck for good
        process.terminate()

        # Kill the process
        process.kill()

        process.join()

        # we will use this to re-try running CLASS after adding a jitter term to the parameters
        state = False

        results = None

    else:

        state = True

        results = queue.get()

    return state, results
