# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Setup for the redshift distributions
'''

import numpy as np
import scipy.stats as ss

# our scripts
import settings as st


class nz_dist(object):

    def __init__(self, zmin: float = None, zmax: float = None, nzmax: int = None):

        if zmin is None:
            self.zmin = st.survey_zmin

        else:
            self.zmin = zmin

        if zmax is None:
            self.zmax = st.survey_zmax

        else:
            self.zmax = zmax

        if nzmax is None:
            self.nzmax = st.nzmax

        else:
            self.nzmax = nzmax

        # n(z) redshift
        self.nz_z = np.linspace(self.zmin, self.zmax, self.nzmax)

        # shift to midpoint
        self.mid_z = 0.5 * (self.nz_z[1:] + self.nz_z[:-1])

        self.mid_z = np.concatenate((np.zeros(1), self.mid_z))

    def nz_model_1(self, zm: float) -> np.ndarray:
        '''
        Calculate the analytic function

        :math:`n(z)=z^{2}\\text{exp}(-\\frac{z}{z_{0}})`
        '''
        z0 = zm / 3.

        nz = self.nz_z**2 * np.exp(-self.nz_z / z0)

        # find normalisation factor
        fact = np.trapz(nz, self.nz_z)

        # calculate n(z) at mid points and normalise n(z)
        nz_new = self.mid_z**2 * np.exp(-self.mid_z / z0) / fact

        return nz_new

    def nz_model_2(self, z0: float, alpha: float = 2, beta: float = 1.5):
        '''
        https://arxiv.org/pdf/1502.05872.pdf

        Calculate the analytic function

        :math:`n(z)=z^{\\alpha}\\text{exp}(-(\\frac{z}{z_{0}})^{\\beta})`
        '''

        nz = self.nz_z**alpha * np.exp(-(self.nz_z / z0)**beta)

        # find normalisation factor
        fact = np.trapz(nz, self.nz_z)

        # calculate n(z) at mid points and normalise n(z)
        nz_new = self.mid_z**alpha * np.exp(-(self.mid_z / z0)**beta) / fact

        return nz_new

    def nz_gaussian(self, z0: float, sigma: float) -> np.ndarray:
        '''
        Gaussian n(z) distribution for the tomographic bin

        :math:`n(z)=\\frac{1}{2\pi\sigma}\\text{exp}(-\\frac{1}{2}\\frac{(z-z_{0})^{2}}{\sigma^{2}})`
        '''

        nz_dist = ss.norm(z0, sigma)

        nz = nz_dist.pdf(self.nz_z)

        return nz
