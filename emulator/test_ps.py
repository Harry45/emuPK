# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Generate k_min - exploratory analysis to understand better the range of k
'''

# import time
# import numpy as np
# import matplotlib.pylab as plt

# # our script
# import cosmoclass.spectrum as sp
# import cosmogp as cgp
# import utils.powerspec as up
# import setemu as st

# plt.rc('text', usetex=True)
# plt.rc('font',**{'family':'sans-serif','serif':['Palatino']})
# figSize  = (12, 8)
# fontSize = 20

# # the cosmology module (will use CLASS here)
# cosmo_module = sp.matterspectrum(zmax=4.66)
# cosmo_module.input_configurations()

# # Central point
# point = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 0.5692])

# start_time = time.time()
# pk_class = cosmo_module.interpolated_spectrum(point, 'cubic')
# end_time = time.time()
# print("3D matter power by CLASS: {0:.3f} seconds".format(end_time - start_time))


# ps_gp = cgp.gp_power_spectrum(zmax=4.66)
# ps_gp.input_configurations()
# ps_gp.load_gps('gps/')

# start_time = time.time()
# pk_gp = ps_gp.interpolated_spectrum(point, 'cubic')
# end_time = time.time()
# print("3D matter power spectrum by GP: {0:.3f} seconds".format(end_time - start_time))


# gp_lin = ps_gp.pk_lin(point)
# co_lin = cosmo_module.pk_lin(point)

# plt.figure(figsize = figSize)
# plt.loglog(ps_gp.k_new, co_lin, basex=10, basey=10, lw = 2, label = 'CLASS')
# plt.loglog(ps_gp.k_new, gp_lin, basex=10, basey=10, lw = 2, linestyle='--', label = 'Emulator')
# plt.ylim(2E-2, 60E3)
# plt.ylabel(r'$P_{\delta}(k,z=0$', fontsize = fontSize)
# plt.xlabel(r'$k[h\,\textrm{Mpc}^{-1}]$', fontsize = fontSize)
# plt.tick_params(axis='x', labelsize=fontSize)
# plt.tick_params(axis='y', labelsize=fontSize)
# plt.legend(loc = 'best',prop={'family':'sans-serif', 'size':15})
# #plt.savefig('/home/harry/Desktop/test/pk_'+str(z_index)+'.png', bbox_inches = 'tight', dpi = 100)
# plt.show()


# Gradient Test
# eps = 1E-5
# point_p = np.array([0.1295 + eps, 0.0224, 2.895, 0.9948, 0.7411, 0.5692, 1.0078])
# point_m = np.array([0.1295 - eps, 0.0224, 2.895, 0.9948, 0.7411, 0.5692, 1.0078])

# pk_p = cosmo_module.compute_ps(point_p)[0, 0]
# pk_m = cosmo_module.compute_ps(point_m)[0, 0]

# print(pk[0, 0])
# print(pk_p)
# print(pk_m)
# print((pk_p - pk_m) / (2 * eps))


# # gradients
# start_time = time.time()
# grad_gp = ps_gp.interp_gradient(point, 1, 'cubic')
# end_time = time.time()
# print("Gradient 3D matter power spectrum by GP: {0:.3f} seconds".format(end_time - start_time))

# start_time = time.time()
# grad_class = cosmo_module.interpolated_gradient(point)
# end_time = time.time()
# print("Gradient 3D matter power by CLASS: {0:.3f} seconds".format(end_time - start_time))


# for i in range(7):

# 	fig, ax = plt.subplots(figsize = figSize)
# 	plt.plot(cosmo_module.k_new, grad_class[i][15], lw = 2, label = 'CLASS')
# 	plt.plot(ps_gp.k_new, grad_gp[i][15], lw = 2, linestyle='--', label = 'Emulator')
# 	plt.xlim(st.k_min_h_by_Mpc, st.k_max_h_by_Mpc)
# 	plt.xscale('log')
# 	plt.ylabel(r'$\frac{\partial P_{\delta}(k,z=0)}{\partial\theta_{'+str(i)+'}}$', fontsize = fontSize)
# 	plt.xlabel(r'$k[h\,\textrm{Mpc}^{-1}]$', fontsize = fontSize)
# 	plt.tick_params(axis='x', labelsize=fontSize)
# 	plt.tick_params(axis='y', labelsize=fontSize)
# 	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# 	ax.yaxis.offsetText.set_fontsize(fontSize)
# 	plt.legend(loc = 'best',prop={'family':'sans-serif', 'size':15})
# 	plt.savefig('/home/harry/Desktop/Video-Power-Spectra/gradients/grad_'+str(i)+'.pdf', bbox_inches = 'tight', dpi = 100)
# 	plt.close()

# for z_index in range(100):

# 	plt.figure(figsize = figSize)
# 	plt.loglog(ps_gp.k_new, pk_class[z_index], basex=10, basey=10, lw = 2, label = 'CLASS')
# 	plt.loglog(ps_gp.k_new, pk_gp[z_index], basex=10, basey=10, lw = 2, linestyle='--', label = 'Emulator')
# 	plt.ylim(2E-2, 60E3)
# 	plt.ylabel(r'$P_{\delta}(k,z='+str(np.around(ps_gp.z_new[z_index],2))+')$', fontsize = fontSize)
# 	plt.xlabel(r'$k[h\,\textrm{Mpc}^{-1}]$', fontsize = fontSize)
# 	plt.tick_params(axis='x', labelsize=fontSize)
# 	plt.tick_params(axis='y', labelsize=fontSize)
# 	plt.legend(loc = 'best',prop={'family':'sans-serif', 'size':15})
# 	plt.savefig('/home/harry/Desktop/test/pk_'+str(z_index)+'.png', bbox_inches = 'tight', dpi = 100)
# 	plt.close()


import utils.helpers as hp
import optimisation as op
import numpy as np
np.set_printoptions(suppress=True, precision=3)

# [201.52747821 850.89656834 201.52747833 201.52747831 851.16967761]

inputs = hp.load_arrays('processing/trainingpoints', 'scaled_inputs')
outputs = hp.load_arrays('processing/trainingpoints/out_class_q', 'q_798')
point = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 0.5692])
gp = op.maximise(inputs, outputs)

print(gp.prediction(point))
print(gp.opt_params)
print(gp.pred_original_function(point))

# [47.505 46.552 45.529 44.407 43.173 41.824 40.342 38.719 37.04  35.36 ]

# gps_q = [hp.load_pkl_file('gps/out_class_q', 'gp_q_' + str(i)) for i in range(800)]
# for gp in gps_q:
# 	print(gp.opt_params)


# # our script
# import gp_moped as gpm


# # the noise level (jitter term) for the GP is in log
# sigma       = [-40.0]
# inputData   = np.c_[inputs, outputs.reshape(1000, 1)]
# ndim   = 6

# # Set some bounds on the kernel hyperparameters
# # Effectively a prior as we would have done with a full Bayesian analysis
# bnd    = np.repeat(np.array([[-1.5,6]]), ndim+1, axis = 0)
# bnd[0] = np.array([-1.5, 2])


# gp = gpm.GP(inputData, sigma=sigma, train = True, nrestart = 5, ndim = ndim)
# gp.transform()
# gp.fit(method = 'L-BFGS-B', bounds = bnd, options = {'ftol':1E-20, 'maxiter':600})
# mean_ = gp.sampleBandPower(point, mean = True)

# print(mean_)
