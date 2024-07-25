import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

from models import my_system as ts
DATA_PATH = '../../data/water_wave/'

def main():
	"""
	This program plots the velocity of a negatively buoyant inertial particle
	in a linear water wave over time.
	"""
	# initialize variables
	x_0, z_0 = 0, 0
	St, beta = 0.01, 0.9
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	R = 2 / 3 * beta

	# read numerical data
	numerics = pd.read_csv('../data/water_wave/numerics.csv')

	# create condition and lambdas to help filter through numerical data
	condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
									& (numerics['St'] == St) \
									& (numerics['beta'] == beta) \
									& (numerics['h\''] == h) \
									& (numerics['A\''] == A) \
									& (numerics['wavelength\''] == wavelength) \
									& (numerics['delta_t\''] == delta_t)
	get = lambda name, history : numerics[name].where(condition \
							   & (numerics['history'] == history))\
											   .dropna().to_numpy()
	# retrieve relevant numerical results
	x = get('x', False)
	z = get('z', False)
	xdot = get('xdot', False)
	zdot = get('zdot', False)
	t = get('t', False)
	x_history = get('x', True)
	z_history = get('z', True)
	xdot_history = get('xdot', True)
	zdot_history = get('zdot', True)
	t_history = get('t', True)

	# initialize trajectory figure
	plt.figure(1)
	fs, lfs = 14, 16	# fontsizes
	n = int(np.ceil(len(t) * 0.6))
	plt.xlabel(r'$x$', fontsize=lfs)
	plt.ylabel(r'$z$', fontsize=lfs)
#	plt.gca().set_aspect('equal')
	plt.gca().set_box_aspect(1)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# plot particle trajectories
	plt.plot(x, z, c='k', label='without history effects')
	plt.plot(x_history, z_history, c='k', ls='--', label='with history effects')
#	plt.legend(fontsize=fs)

	# initialize velocity figure and top subplot
	plt.figure(2)
	plt.subplot(211)
	plt.ylabel(r'$v_x$', fontsize=lfs)
	plt.xticks([])
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# plot horizontal numerical results
	plt.plot(t[:n], xdot[:n], c='k', label='without history effects')
	plt.plot(t_history[:n], xdot_history[:n], c='k', ls=':',
			 label='with history effects')
#	plt.legend(fontsize=fs)

	# initialize bottom subplot
	plt.subplot(212)
	plt.xlabel(r'$t$', fontsize=lfs)
	plt.ylabel(r'$v_z$', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# plot vertical numerical results
	plt.plot(t[:n], zdot[:n], c='k', label='without history effects')
	plt.plot(t_history[:n], zdot_history[:n], c='k', ls=':',
			 label='with history effects')
	plt.show()

if __name__ == '__main__':
	main()
