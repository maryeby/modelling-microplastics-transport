import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '../../data/water_wave/'

def main():
	"""
	This program plots numerical and analytical solutions for the Stokes drift
	velocity of a negatively buoyant inertial particle in a linear water wave.
	"""
	# read data
	numerics = pd.read_csv(f'{DATA_PATH}numerics.csv')
	analysis = pd.read_csv(f'{DATA_PATH}santamaria_fig2_recreation.csv')

	# initialize drift velocity figure & left subplot
	plt.figure(1)
	plt.suptitle(r'Santamaria Figure 2 Recreation', fontsize=18)
	plt.subplot(121)
	plt.xlabel(r'$\omega t$', fontsize=16)
	plt.ylabel(r'$\frac{u_d}{U\mathrm{Fr}}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# retrieve relevant numerical results
	get_analysis = lambda name, history : analysis[name].where(\
											analysis['history'] == history)\
														.dropna()
	u_d = get_analysis('u_d', False)
	t = get_analysis('t', False)
	u_d_history = get_analysis('u_d', True)
	t_history = get_analysis('t', True)

	# plot horizontal numerical results
	plt.scatter(t, u_d, marker='o', edgecolors='k', facecolors='none',
				label='numerics without history')
	plt.scatter(t_history, u_d_history, c='k', marker='x',
				label='numerics with history')
	plt.legend(fontsize=14)

	# initialize right subplot
	plt.subplot(122)
	plt.xlabel(r'$\omega t$', fontsize=16)
	plt.ylabel(r'$\frac{w_d}{g \tau}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# retrieve relevant numerical results
	w_d = get_analysis('w_d', False)
	t = get_analysis('t', False)
	w_d_history = get_analysis('w_d', True)
	t_history = get_analysis('t', True)

	# plot vertical numerical results
	plt.scatter(t, w_d, edgecolors='k', facecolors='none', marker='o',
				label='numerics without history')
	plt.scatter(t_history, w_d_history, c='k', marker='x',
				label='numerics with history')
	plt.legend(fontsize=14)

	# initialize particle trajectory figure
	plt.figure(2)
	plt.title(r'Particle Trajectory', fontsize=18)
	plt.xlabel('x', fontsize=14)
	plt.ylabel('z', fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# create conditions to help filter through numerical data
	x_0, z_0 = 0, 0
	St, beta = 0.01, 0.9
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
									 & (numerics['St'] == St) \
									 & (numerics['beta'] == beta) \
									 & (numerics['h\''] == h) \
									 & (numerics['A\''] == A) \
									 & (numerics['wavelength\''] == wavelength)\
									 & (numerics['delta_t\''] == delta_t)
	get_numerics = lambda name, history : numerics[name].where(condition \
											& (numerics['history'] == history))\
														.dropna()
	# retrieve relevant numerical results
	x = get_numerics('x', False)
	z = get_numerics('z', False)
	x_history = get_numerics('x', True)
	z_history = get_numerics('z', True)

	x_crossings = get_analysis('x_crossings', False)
	z_crossings = get_analysis('z_crossings', False)

	# plot particle trajectory with and without history
	plt.plot(x, z, c='k', label='without history')
	plt.plot(x_history, z_history, c='k', ls='--', label='with history')
	plt.scatter(x_crossings, z_crossings, c='k', marker='x',
				label='zero crossings')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
