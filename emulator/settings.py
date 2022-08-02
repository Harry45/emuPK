# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

# -----------------------------------------------------------------------------

# method for building the emulator
# "components" refers to the fact that we are using the:
# - linear matter power spectrum, P_lin(k,z0)
# - growth factor, A(z)
# - q function, q(k,z)
# to reconstruct the non-linear matter power spectrum as
# p_nonlinear = A(z) * [1 + q(k,z)] * P_lin(k,z0)

components = True

# if we want to include sample neutrino mass
# this does not imply that neutrino is not included in CLASS
neutrino = False

if not neutrino:

    # fixed neutrino mass
    fixed_nm = {'M_tot': 0.06}

# time out function (in seconds)
# CLASS does not run for certain cosmologies
timeout = 60

timed = True

# -----------------------------------------------------------------------------
# Important parameter inputs for calculating the matter power spectrum

# mode for calculating the power spectrum
# 'halofit' or 'hmcode'

mode = 'halofit'

# settings for halofit

# halofit needs to evaluate integrals (linear power spectrum times some
# kernels). They are sampled using this logarithmic step size

halofit_k_per_decade = 80.  # default in CLASS is 80

# a smaller value will lead to a more precise halofit result at the
# highest redshift at which halofit can make computations,at the expense
# of requiring a larger k_max; but this parameter is not relevant for the
# precision on P_nl(k,z) at other redshifts, so there is normally no need
# to change it

halofit_sigma_precision = 0.05  # default in CLASS is 0.05


# settings for hmcode

# cmin =

# eta =

# minimum redshift
zmin = 0.0

# maximum redshift
zmax = 4.66

# maximum of k (for quick CLASS run, set to for example, 50)
k_max_h_by_Mpc = 50.

# our wanted kmax
kmax = 50.0

# minimum of k
k_min_h_by_Mpc = 5E-4

# number of k
nk = 40

# number of redshift on the grid
nz = 20

# new number of k (interpolated)
nk_new = 1000

# new number of z (interpolated)
nz_new = 100

# curvature
Omega_k = 0.

# pivot scale in $ Mpc^{-1}$
k_pivot = 0.05

# Big Bang Nucleosynthesis
bbn = '/home/harry/Desktop/class/bbn/sBBN.dat'

# -----------------------------------------------------------------------------

# Priors
# specs are according to the scipy.stats. See documentation:
# https://docs.scipy.org/doc/scipy/reference/stats.html

# For example, if we want uniform prior between 1.0 and 5.0, then
# it is specified by loc and loc + scale, where scale=4.0
# distribution = scipy.stats.uniform(1.0, 4.0)

priors = {

    'omega_cdm': {'distribution': 'uniform', 'specs': [0.06, 0.34]},
    'omega_b': {'distribution': 'uniform', 'specs': [0.019, 0.007]},
    'ln10^{10}A_s': {'distribution': 'uniform', 'specs': [1.70, 3.30]},
    'n_s': {'distribution': 'uniform', 'specs': [0.70, 0.60]},
    'h': {'distribution': 'uniform', 'specs': [0.64, 0.18]}
}

# choose which cosmological parameters to marginalise over
# first 5 are by default
if neutrino:

    # list of cosmological parameters to use
    # we suggest keeping this order since the emulator inputs are in the same order
    cosmology = ['omega_cdm', 'omega_b', 'ln10^{10}A_s', 'n_s', 'h', 'M_tot']

    # add prior for neutrino
    priors['M_tot'] = {'distribution': 'uniform', 'specs': [0.01, 0.99]}

else:
    cosmology = ['omega_cdm', 'omega_b', 'ln10^{10}A_s', 'n_s', 'h']

# -----------------------------------------------------------------------------
# Baryon Feedback settings

# baryon model to be used
baryon_model = 'AGN'

cst = {'AGN': {'A2': -0.11900, 'B2': 0.1300, 'C2': 0.6000, 'D2': 0.002110, 'E2': -2.0600,
               'A1': 0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1': 1.8400,
               'A0': 0.15000, 'B0': 1.2200, 'C0': 1.3800, 'D0': 0.001300, 'E0': 3.5700},
       'REF': {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2': 0.8580,
               'A1': 0.07280, 'B1': 0.0381, 'C1': 1.0600, 'D1': 0.006520, 'E1': -1.7900,
               'A0': 0.00972, 'B0': 1.1200, 'C0': 0.7500, 'D0': -0.000196, 'E0': 4.5400},
       'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2': 0.001990, 'E2': -0.8250,
                 'A1': 0.49000, 'B1': 0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                 'A0': -0.01660, 'B0': 1.0500, 'C0': 1.3000, 'D0': 0.001200, 'E0': 4.4800}}


# -----------------------------------------------------------------------------

# Settings for the GP emulator module

# noise/jitter term
var = 1E-5

# another jitter term for numerical stability
jitter = 1E-5

# order of the polynomial (maximum is 2)
order = 2

# Transform input (pre-whitening)
x_trans = True

# Centre output on 0 if we want
use_mean = False

# Number of times we want to restart the optimiser
n_restart = 5

# minimum lengthscale (in log)
l_min = -5.0

# maximum lengthscale (in log)
l_max = 5.0

# minimum amplitude (in log)
a_min = 0.0

# maximum amplitude (in log)
a_max = 25.0

# choice of optimizer (better to use 'L-BFGS-B')
method = 'L-BFGS-B'

# tolerance to stop the optimizer
ftol = 1E-30

# maximum number of iterations
maxiter = 600

# decide whether we want to delete the kernel or not
del_kernel = True

# growth factor (not very broad distribution in function space)
gf_args = {'y_trans': False, 'lambda_cap': 1}

# if we want to emulate 1 + q(k,z):
emu_one_plus_q = False

if emu_one_plus_q:

    # q function (expected to be zero)
    qf_args = {'y_trans': True, 'lambda_cap': 1}

    # folder where we will store the files
    d_one_plus = '_op'

else:

    # q function (expected to be zero)
    qf_args = {'y_trans': False, 'lambda_cap': 1}

    # folder where we will store the files
    d_one_plus = ''

# linear matter power spectrum
pl_args = {'y_trans': True, 'lambda_cap': 1000}

# non linear matter power spectrum
pknl_args = {'y_trans': True, 'lambda_cap': 1000}

# -----------------------------------------------------------------------------
# For survey setting

nzmax = 75

survey_zmin = 0.0

survey_zmax = 2.0

nellsmax = 39

ell_min = 10

ell_max = 3500

nell_int = 1000
