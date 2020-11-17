# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Module to load all GPs and predict mean, first derivative and second derivative
'''

import time
import numpy as np

# our script
import utils.helpers as hp
import utils.powerspec as up
import setemu as st

class gp_power_spectrum(object):
    '''
    Predict the 3D matter power spectrum at a given test point in parameter space

    :param: zmax (float) : value of maximum redshift (default is 4.66 - see setemu for frther details)
    '''

    def __init__(self, zmax:float=st.zmax):

        # maximum redshift
        self.zmax = zmax

    def input_configurations(self)->None:
        # redshift range
        self.z = np.linspace(0.0, self.zmax, st.nz, endpoint=True)

        # k range
        self.k_range = np.logspace(np.log10(st.k_min_h_by_Mpc), np.log10(st.k_max_h_by_Mpc), st.nk, endpoint=True)

        # new redshift range
        self.z_new = np.linspace(0.0, self.zmax, st.nz_new, endpoint=True)

        # new k range
        self.k_new = np.logspace(np.log10(st.k_min_h_by_Mpc), np.log10(st.k_max_h_by_Mpc), st.nk_new, endpoint=True)

    def load_gps(self, gp_dir:str)->list:
        '''
        Load all the GPs
        
        :param: gp_dir (str) : directory where the trained GPs are stored

        :return: all_gps (list) : list containing all GPs
        '''
        # the GPs directory
        self.gp_dir = gp_dir

        # total number of GPs
        self.ngps = int(st.nk * st.nz)

        start_time = time.time()

        all_gps = [hp.load_pkl_file(self.gp_dir, 'gp_pk_'+str(i)) for i in range(self.ngps)]

        end_time = time.time()

        print("All GPs loaded in {0:.3f} seconds".format(end_time - start_time))

        self.all_gps = all_gps

        return all_gps

    def mean_pred(self, testpoint:np.ndarray)->np.ndarray:
        '''
        Mean prediction of the GP at a test point in parameter space

        :param: testpoint (np.ndarray) : a testpoint

        :return: results (np.ndarray) : the mean prediction from all GPs
        '''

        arguments = list(zip([testpoint] * self.ngps, self.all_gps))

        results = np.array(list(map(up.prediction, arguments)))

        # reshape the results in the right format 
        results = results.reshape(st.nk, st.nz)

        return results

    def interpolated_spectrum(self, testpoint:np.ndarray, int_type: str = 'cubic')->np.ndarray:
        '''
        Mean prediction of the GP at a test point in parameter space

        :param: testpoint (np.ndarray) : a testpoint

        :param: int_type (str) : type of interpolation (linear, cubic, quintic). Default is cubic

        :return: interpolation (np.ndarray) : the interpolated 3D matter power spectrum
        '''

        # calculate the mean prediction
        pred = self.mean_pred(testpoint).flatten()

        # inputs to the interpolator
        inputs = [self.k_range, self.z, pred, int_type]

        # the new grid
        grid = [self.k_new, self.z_new]

        # the interpolated power spectrum
        spectra = up.kz_interpolate(inputs, grid)

        return spectra


    def gradient(self, testpoint:np.ndarray, order:int = 1)->np.ndarray:
        '''
        Calculate the gradient of the power spectrum

        :param: testpoint (np.ndarray) : a testpoint

        :param: order (int) : 1 or 2 (1 refers to first derivative and 2 refers to second)

        :return: the derivatives of the power spectrum (of size (nk x nz) x ndim), for example 800 x 7 
        '''

        arguments = list(zip([testpoint] * self.ngps, self.all_gps, [order]*self.ngps))

        results = np.array(list(map(up.gradient, arguments)))

        return results

    def interp_gradient(self, testpoint:np.ndarray, order:int = 1, int_type: str = 'cubic'):
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
                inputs = [self.k_range, self.z, first_der[:,p], int_type]

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
    all_gps = ps_gp.load_gps('gps/')

    point = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 0.5692, 1.0078])
    # interp_ps = ps_gp.interpolated_spectrum(point, 'cubic')
    grad = ps_gp.interp_gradient(point, 1, 'cubic')

    import matplotlib.pylab as plt 

    plt.rc('text', usetex=True)
    plt.rc('font',**{'family':'sans-serif','serif':['Palatino']})
    figSize  = (12, 8)
    fontSize = 20

    # plt.figure(figsize = figSize)
    fig, ax = plt.subplots(figsize = figSize)
    plt.plot(ps_gp.k_new, grad[4][0], lw = 2)
    plt.xlim(st.k_min_h_by_Mpc, st.k_max_h_by_Mpc)
    plt.xscale('log')
    plt.ylabel(r'$\frac{\partial P_{\delta}(k,z=0)}{\partial\theta_{0}}$', fontsize = fontSize)
    plt.xlabel(r'$k[h\,\textrm{Mpc}^{-1}]$', fontsize = fontSize)
    plt.tick_params(axis='x', labelsize=fontSize)
    plt.tick_params(axis='y', labelsize=fontSize)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.yaxis.offsetText.set_fontsize(fontSize)
    plt.show()
