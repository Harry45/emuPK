# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development


'''
Calculate and store the predictions at test points in parameter space
'''

from typing import Tuple
import numpy as np

# our Python scripts
import cosmology.spectrumcalc as sc
import settings as st
import utils.helpers as hp
import priors as pr
import cosmology.cosmofuncs as cf


def main(ntest: int = 1, a_bary: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate the non-linear matter power spectrum at test points in parameter space

    :param: ntest (int) - the number of test points we want to use

    :param: a_bary (float) - the Baryon Feedback parameter (default: 1.0)

    :return: pk_class (np.ndarray) - the 3D power spectrum calculated by CLASS

    :return: pk_gp (np.ndarray) - the 3D power spectrum calculated by the emulator
    '''

    # set the prior first
    dists = pr.all_entities(st.priors)

    # CLASS object
    class_model = sc.matterspectrum(emulator=False)

    # GPs
    gp_model = sc.matterspectrum(emulator=True)

    # additional line to load all the GPs (no need for this if emulator=False - falls back to CLASS)
    gp_model.load_gps(directory='semigps')

    # create empty list to store all the information
    cl_rec = []
    gp_rec = []

    for i in range(ntest):

        # generate a parameter from the prior space
        par = [dists[x].rvs() for x in st.cosmology]

        # generate the dictionary
        par_dict = cf.mk_dict(st.cosmology, par)

        # calculate the non-linear matter power spectrum with CLASS and GP
        gp_gf, gp_pk_nl, gp_pk_l = gp_model.int_pk_nl(par_dict, a_bary)
        cl_gf, cl_pk_nl, cl_pk_l = class_model.int_pk_nl(par_dict, a_bary)

        gp_rec.append(gp_pk_nl)
        cl_rec.append(cl_pk_nl)

    cl_rec = np.asarray(cl_rec)
    gp_rec = np.asarray(gp_rec)

    # save the Pk
    hp.store_arrays(gp_rec, 'predictions', 'gp_pk' + st.d_one_plus)
    hp.store_arrays(cl_rec, 'predictions', 'cl_pk' + st.d_one_plus)

    return cl_rec, gp_rec


if __name__ == "__main__":
    cl_rec, gp_rec = main(ntest=100, a_bary=1.0)
