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
	analysis = pd.read_csv(f'{DATA_PATH}velocity_analysis.csv')

	# initialize drift velocity figure & left subplot
	plt.figure()
	plt.subplot(121)
	plt.xlabel(r'$t$', fontsize=16)
	plt.ylabel(r'$u_d$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# retrieve relevant numerical results
	get = lambda name, history : analysis[name].where(analysis['history']
													  == history).dropna()
	u_d = get('u_d', False)
	t = get('t', False)
	fitted_t_u = get('fitted_t_u', False)
	u_d_history = get('u_d', True)
	t_history = get('t', True)
	fitted_t_u_history = get('fitted_t_u', True)

	# plot horizontal numerical results
	plt.scatter(t, u_d, marker='o', edgecolors='k', facecolors='none',
				label='numerics without history')
	plt.scatter(t_history, u_d_history, marker='s', edgecolors='k',
				facecolors='none', label='numerics with history')
#	plt.plot(fitted_t_u, u_d, c='k', ls='--', label='fitted without history')
#	plt.plot(fitted_t_u_history, u_d_history, c='k',
#			 label='fitted with history')

	# initialize right subplot
	plt.subplot(122)
	plt.xlabel(r'$t$', fontsize=16)
	plt.ylabel(r'$w_d$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# retrieve relevant numerical results
	w_d = get('w_d', False)
	t = get('t', False)
	fitted_t_w = get('fitted_t_w', False)
	w_d_history = get('w_d', True)
	t_history = get('t', True)
	fitted_t_w_history = get('fitted_t_w', True)

	# plot vertical numerical results
	plt.scatter(t, w_d, edgecolors='k', facecolors='none', marker='o',
				label='numerics without history')
	plt.scatter(t_history, w_d_history, edgecolors='k', facecolors='none',
				marker='s', label='numerics with history')
#	plt.plot(fitted_t_w, w_d, c='k', ls='--', label='fitted without history')
#	plt.plot(fitted_t_w_history, w_d_history, c='k',
#			 label='fitted with history')
	plt.show()

if __name__ == '__main__':
	main()
