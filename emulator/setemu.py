# -----------------------------------------------------------------------------

# Important parameter inputs for calculating the matter power spectrum

include_feedback = True

# mode for calculating the power spectrum+
mode = 'halofit'

# baryon model to be used
baryon_model = 'AGN'

# maximum redshift
zmax = 4.66

# maximum of k
k_max_h_by_Mpc = 50.

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

# ell min
ell_min = 10

# ell_max
ell_max = 4001

# -----------------------------------------------------------------------------

# Priors
# specs are according to the scipy.stats. See documentation:
# https://docs.scipy.org/doc/scipy/reference/stats.html

# For example, if we want uniform prior between 1.0 and 5.0, then
# it is specified by loc and loc + scale, where scale=4.0
# distribution = scipy.stats.uniform(1.0, 4.0)

# First seven parameters are for the emulator
p1 = {'distribution': 'uniform', 'parameter': 'omega_cdm', 'specs': [0.01, 0.39]}
p2 = {'distribution': 'uniform', 'parameter': 'omega_b', 'specs': [0.019, 0.007]}
p3 = {'distribution': 'uniform', 'parameter': 'ln10^{10}A_s', 'specs': [1.70, 3.30]}
p4 = {'distribution': 'uniform', 'parameter': 'n_s', 'specs': [0.70, 0.60]}
p5 = {'distribution': 'uniform', 'parameter': 'h', 'specs': [0.64, 0.18]}
p6 = {'distribution': 'uniform', 'parameter': 'sum_neutrino', 'specs': [0.06, 0.94]}
p7 = {'distribution': 'uniform', 'parameter': 'A_bary', 'specs': [0.0, 2.0]}

emu_params = [p1, p2, p3, p4, p5, p6, p7]

# -----------------------------------------------------------------------------
# Baryon Feedback settings

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

# width of the Gaussian Prior (parametric part)
lambda_cap = 1000

# order of the polynomial (maximum is 2)
order = 2

# Transform input (pre-whitening)
x_trans = True

# Transform output (logarithm transform - log10)
y_trans = True

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
ftol = 1E-20

# maximum number of iterations
maxiter = 500

# decide whether we want to delete the kernel or not
del_kernel = True

# -----------------------------------------------------------------------------

# some tests for spectrum routine

# k_int = np.linspace(st.k_min_by_Mpc, st.k_max_by_Mpc, 5000)
# ps_int = ut.ps_interpolate([self.k_range, pk_matter[:,10], k_int])

# plt.figure(figsize = figSize)
# plt.loglog(k_int, ps_int, basex=10, basey=10)
# plt.ylabel(r'$P_{\delta}(k,z)$', fontsize = fontSize)
# plt.xlabel(r'$k\left[\textrm{Mpc}^{-1}\right]$', fontsize = fontSize)
# plt.tick_params(axis='x', labelsize=fontSize)
# plt.tick_params(axis='y', labelsize=fontSize)
# plt.show()


# if __name__ == "__main__":
#     inputs = hp.load_arrays('processing/trainingpoints', 'scaled_inputs')
#     outputs = hp.load_arrays('processing/trainingpoints/processed_pk', 'pk_0')

#     gp = train_gp(inputs, outputs)

#     point = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 0.5692, 1.0078])
#     print(gp.pred_original_function(point))
#     print(gp.derivatives(point, order=2))

