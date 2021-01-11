# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Module to load all GPs and predict mean, first derivative and second derivative
'''

import os
import time
from typing import Tuple
import numpy as np

# our script
import utils.helpers as hp
import utils.powerspec as up
import setemu as st
np.set_printoptions(suppress=True, precision=3)


class gp_power_spectrum(object):
    '''
    Predict the 3D matter power spectrum at a given test point in parameter space

    :param: zmax (float) : value of maximum redshift (default is 4.66 - see setemu for frther details)
    '''

    def __init__(self, zmax: float = st.zmax):

        # maximum redshift
        self.zmax = zmax

    def input_configurations(self) -> None:
        # redshift range
        self.z = np.linspace(0.0, self.zmax, st.nz, endpoint=True)

        # k range
        self.k_range = np.logspace(np.log10(st.k_min_h_by_Mpc), np.log10(st.k_max_h_by_Mpc), st.nk, endpoint=True)

        # new redshift range
        self.z_new = np.linspace(0.0, self.zmax, st.nz_new, endpoint=True)

        # new k range
        self.k_new = np.logspace(np.log10(st.k_min_h_by_Mpc), np.log10(st.k_max_h_by_Mpc), st.nk_new, endpoint=True)

    def load_gps(self, gp_dir: str) -> list:
        '''
        Load all the GPs

        :param: gp_dir (str) : directory where the trained GPs are stored

        :return: all_gps (list) : list containing all GPs
        '''
        if st.gf_class:
            self.ng = len(os.listdir(gp_dir + '/out_class_g'))
            self.nl = len(os.listdir(gp_dir + '/out_class_l'))
            self.nq = len(os.listdir(gp_dir + '/out_class_q'))

            start_time = time.time()

            self.gps_g = [hp.load_pkl_file(gp_dir + '/out_class_g', 'gp_g_' + str(i)) for i in range(self.ng)]
            self.gps_l = [hp.load_pkl_file(gp_dir + '/out_class_l', 'gp_l_' + str(i)) for i in range(self.nl)]
            self.gps_q = [hp.load_pkl_file(gp_dir + '/out_class_q', 'gp_q_' + str(i)) for i in range(self.nq)]

            end_time = time.time()

            print("All GPs loaded in {0:.3f} seconds".format(end_time - start_time))

    def pk_lin(self, testpoint: np.ndarray) -> np.ndarray:
        '''
        Calclate the linear matter power spectrum for a given test point

        :param: testpoint (np.ndarray) : a test point in parameter space

        :return: ps (np.ndarray) : the linear matter power spectrum
        '''
        args_l = list(zip([testpoint] * self.nl, self.gps_l))

        results_l = np.array(list(map(up.prediction, args_l)))

        # inputs to the interpolator
        inputs = [self.k_range, results_l, self.k_new]

        ps = up.ps_interpolate(inputs)

        return ps

    def growth_function(self, testpoint: np.ndarray) -> np.ndarray:
        '''
        Calculate the growth function for a given test point

        :param: testpoint (np.ndarray) : a test point in parameter space

        :return: ps (np.ndarray) : the growth function
        '''

        args_g = list(zip([testpoint] * self.ng, self.gps_g))

        results_g = np.array(list(map(up.prediction, args_g)))

        return results_g

    def mean_pred(self, testpoint: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Mean prediction of the GP at a test point in parameter space

        :param: testpoint (np.ndarray) : a testpoint

        :return: results (np.ndarray) : the mean prediction from all GPs
        '''

        args_g = list(zip([testpoint] * self.ng, self.gps_g))
        args_l = list(zip([testpoint] * self.nl, self.gps_l))
        args_q = list(zip([testpoint] * self.nq, self.gps_q))

        results_g = np.array(list(map(up.prediction, args_g)))
        results_l = np.array(list(map(up.prediction, args_l)))
        results_q = np.array(list(map(up.prediction, args_q)))

        return results_g, results_l, results_q

    def matter_ps(self, g_func: np.ndarray, l_ps: np.ndarray, q_func: np.ndarray) -> np.ndarray:
        '''
        Given the growth function, linear matter power spectrum and the non-linear function, we calculate the 3D matter
        power spectrum. Note that this is not the interpolated power spectrum, that is, the above mentioned quantities
        are first calculated on a grid.

        :param: g (np.ndarray) : the growth function

        :param: l (np.ndarray) : the linear matter power spectrum

        :param: q (np.ndarray) : the non-linear function

        :return: p_matter (np.ndarray) : the 3D non-linear matter power spectrum
        '''
        g_func = g_func.reshape(1, st.nz)
        l_ps = l_ps.reshape(st.nk, 1)
        q_func = q_func.reshape(st.nk, st.nz)

        p_matter = g_func * (1. + q_func) * l_ps

        return p_matter

    def interpolated_spectrum(self, testpoint: np.ndarray, int_type: str = 'cubic') -> np.ndarray:
        '''
        Mean prediction of the GP at a test point in parameter space

        :param: testpoint (np.ndarray) : a testpoint

        :param: int_type (str) : type of interpolation (linear, cubic, quintic). Default is cubic

        :return: interpolation (np.ndarray) : the interpolated 3D matter power spectrum
        '''

        # calculate the mean prediction
        g, l, q = self.mean_pred(testpoint)
        print(q[-10:])

        pred = self.matter_ps(g, l, q).flatten()

        # inputs to the interpolator
        inputs = [self.k_range, self.z, pred, int_type]

        # the new grid
        grid = [self.k_new, self.z_new]

        # the interpolated power spectrum
        spectra = up.kz_interpolate(inputs, grid)

        return spectra

    def gradient(self, testpoint: np.ndarray, order: int = 1) -> np.ndarray:
        '''
        Calculate the gradient of the power spectrum

        :param: testpoint (np.ndarray) : a testpoint

        :param: order (int) : 1 or 2 (1 refers to first derivative and 2 refers to second)

        :return: the derivatives of the power spectrum (of size (nk x nz) x ndim), for example 800 x 7
        '''

        arguments = list(zip([testpoint] * self.ngps, self.all_gps, [order] * self.ngps))

        results = np.array(list(map(up.gradient, arguments)))

        return results

    def interp_gradient(self, testpoint: np.ndarray, order: int = 1, int_type: str = 'cubic'):
        '''
        Interpolate gradients along k and z axes for each parameter

        :param: testpoint (np.ndarray) : a testpoint

        :param: order (int) : 1 or 2 (1 refers to first derivative and 2 refers to second)

        :param: int_type (str) : type of interpolation (linear, cubic, quintic). Default is cubic

        :return: grad_1 or grad_2 the (interpolated) derivatives
        '''

        # calculate the mean prediction
        if order == 1:
            first_der = self.gradient(testpoint, order)

            # create an empty list to record interpolated gradient
            grad_1 = []

            for p in range(len(testpoint)):

                # inputs to the interpolator
                inputs = [self.k_range, self.z, first_der[:, p], int_type]

                # the new grid
                grid = [self.k_new, self.z_new]

                # the interpolated power spectrum
                grad = up.kz_interpolate(inputs, grid)

                grad_1.append(grad)

            return grad_1

        else:
            first_der, second_der = self.gradient(testpoint, order)

            # create an empty list to record interpolated gradient
            grad_1, grad_2 = [], []

            return grad_1, grad_2


if __name__ == "__main__":

    ps_gp = gp_power_spectrum(zmax=4.66)
    ps_gp.input_configurations()
    ps_gp.load_gps('gps/')

    point = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 0.5692])
    interp_ps = ps_gp.interpolated_spectrum(point, 'cubic')

    # grad = ps_gp.interp_gradient(point, 1, 'cubic')

    # import matplotlib.pylab as plt

    # plt.rc('text', usetex=True)
    # plt.rc('font', **{'family': 'sans-serif', 'serif': ['Palatino']})
    # figSize = (12, 8)
    # fontSize = 20

    # # plt.figure(figsize = figSize)
    # fig, ax = plt.subplots(figsize=figSize)
    # plt.plot(ps_gp.k_new, grad[4][0], lw=2)
    # plt.xlim(st.k_min_h_by_Mpc, st.k_max_h_by_Mpc)
    # plt.xscale('log')
    # plt.ylabel(r'$\frac{\partial P_{\delta}(k,z=0)}{\partial\theta_{0}}$', fontsize=fontSize)
    # plt.xlabel(r'$k[h\,\textrm{Mpc}^{-1}]$', fontsize=fontSize)
    # plt.tick_params(axis='x', labelsize=fontSize)
    # plt.tick_params(axis='y', labelsize=fontSize)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax.yaxis.offsetText.set_fontsize(fontSize)
    # plt.show()
