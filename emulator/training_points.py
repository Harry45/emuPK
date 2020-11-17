# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Generate the set of training points to be used in the emulator
'''

import numpy as np

# our scripts
import cosmoclass.spectrum as sp
import utils.helpers as hp

# load input training points
inputs = hp.load_arrays('processing/trainingpoints', 'scaled_inputs')
npoints = inputs.shape[0]

# get the cosmology module
cosmo_module = sp.matterspectrum(zmax=4.66)
cosmo_module.input_configurations()

for i in range(npoints):
    pk = cosmo_module.compute_ps(inputs[i])
    hp.store_arrays(pk, 'processing/trainingpoints/pk', 'pk_' + str(i))
