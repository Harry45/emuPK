# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Generate k_min - exploratory analysis to understand better the range of k
'''

import numpy as np
import pandas as pd

import utils.helpers as hp
import cosmoclass.spectrum as sp

# load input training points
inputs = hp.load_arrays('processing/trainingpoints', 'scaled_inputs')

# number of training points
npoints = inputs.shape[0]

# create an empty array to store k_min
k_min_vals = np.zeros(npoints)

# get the cosmology module
cosmo_module = sp.matterspectrum(zmax=4.66)
cosmo_module.input_configurations()

# calculate k_min
for i in range(npoints):
    k_min_vals[i] = cosmo_module.compute_k_min(inputs[i])

# stack the k_min values with the inputs
k_min_output = np.c_[inputs, np.atleast_2d(k_min_vals).T]

# create a pandas dataframe - which we will store
df = pd.DataFrame(k_min_output)

# save to an excel file
hp.save_excel(df, 'processing/kmin', 'kminvalues')
