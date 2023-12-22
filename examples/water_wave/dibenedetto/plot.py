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
	history = analysis['history'] == True
	no_history = analysis['history'] == False

	u_d = analysis['u_d'].where(no_history).dropna()
	t = analysis['t'].where(no_history).dropna()
	u_d_history = analysis['u_d'].where(history).dropna()
	t_history = analysis['t'].where(history).dropna()

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
	w_d = analysis['w_d'].where(no_history).dropna()
	t = analysis['t'].where(no_history).dropna()
	w_d_history = analysis['w_d'].where(history).dropna()
	t_history = analysis['t'].where(history).dropna()

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
	St, beta = 0.01, 0.9
	h, A, wavelength = 50, 0.02, 1 # wave parameters
	delta_t = 5e-3
	history = (numerics['St'] == St) & (numerics['beta'] == beta) \
									 & (numerics['history'] == True) \
									 & (numerics['h'] == h) \
									 & (numerics['A'] == A) \
									 & (numerics['wavelength'] == wavelength) \
									 & (numerics['delta_t'] == delta_t)
	no_history = (numerics['St'] == St) & (numerics['beta'] == beta) \
									& (numerics['history'] == False) \
									& (numerics['h'] == h) \
									& (numerics['A'] == A) \
									& (numerics['wavelength'] == wavelength) \
									& (numerics['delta_t'] == delta_t)

	# retrieve relevant numerical results
	x = numerics['x'].where(no_history).dropna()
	z = numerics['z'].where(no_history).dropna()
	x_history = numerics['x'].where(history).dropna()
	z_history = numerics['z'].where(history).dropna()

	history = analysis['history'] == True
	no_history = analysis['history'] == False
	x_crossings = analysis['x_crossings'].where(no_history).dropna()
	z_crossings = analysis['z_crossings'].where(no_history).dropna()

	# plot particle trajectory with and without history
	plt.plot(x, z, c='k', label='without history')
	plt.plot(x_history, z_history, c='k', ls='--', label='with history')
	plt.scatter(x_crossings, z_crossings, c='k', marker='x',
				label='zero crossings')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
