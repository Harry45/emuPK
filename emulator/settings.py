# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Settings/specifications for the 3D matter power spectrum and aligned with the KiDS-450 analysis
'''

# -----------------------------------------------------------------------------

# KiDS-450 Settings

data_directory = '/home/harry/Dropbox/gp_emulator/data_for_likelihood/'

photoz_method = 'Nz_DIR'

bootstrap_photoz_errors = False  # True #

m_correction = False

index_bootstrap_low = 0

index_bootstrap_high = 999

k_max_h_by_Mpc = 50.

nzmax = 72

nellsmax = 39

mode = 'halofit'

zbin_min = [0.10, 0.30, 0.60]

zbin_max = [0.30, 0.60, 0.90]

bands_EE_to_use = [0, 1, 1, 1, 1, 0, 0]

bands_BB_to_use = [1, 1, 1, 1, 0, 0]

baryon_model = 'AGN'

use_nuisance = ['A_bary', 'A_IA', 'A_noise_z1', 'A_noise_z2', 'A_noise_z3']

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

# the next 4 parameter are nuisance parameters
p8 = {'distribution': 'uniform', 'parameter': 'A_n1', 'specs': [-0.100, 0.200]}
p9 = {'distribution': 'uniform', 'parameter': 'A_n2', 'specs': [-0.100, 0.200]}
p10 = {'distribution': 'uniform', 'parameter': 'A_n3', 'specs': [-0.100, 0.200]}
p11 = {'distribution': 'uniform', 'parameter': 'A_IA', 'specs': [-6.0, 12.0]}

like_params = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]

# -----------------------------------------------------------------------------
