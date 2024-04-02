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
	analysis = pd.read_csv(f'{DATA_PATH}dibenedetto_analysis.csv') # read data

	# initialize drift velocity figure & left subplot
	plt.figure()
	plt.suptitle(r'Time vs Drift Velocity with Dibenedetto Analytics',
				 fontsize=18)
	plt.subplot(121)
	plt.xlabel(r'$t$', fontsize=16)
	plt.ylabel(r'$u_d$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# retrieve relevant numerical and analytical results
	get = lambda name, history : analysis[name].where(\
									analysis['history'] == history).dropna()
	u_d = get('u_d', False)
	t = get('t', False)
	v_x_drift = get('v_x_drift', False)
	u_d_history = get('u_d', True)
	t_history = get('t', True)
	v_x_drift_history = get('v_x_drift', True)

	# plot horizontal numerical results
	plt.plot(t, v_x_drift, c='k', label='analytics without history')
	plt.plot(t_history, v_x_drift_history, c='k', ls='--',
			 label='analytics with history')
	plt.scatter(t, u_d, marker='o', edgecolors='k', facecolors='none',
				label='numerics without history')
	plt.scatter(t_history, u_d_history, c='k', marker='x',
				label='numerics with history')
	plt.legend(fontsize=14)

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
	v_y_drift = get('v_y_drift', False)
	w_d_history = get('w_d', True)
	t_history = get('t', True)
	v_y_drift_history = get('v_y_drift', True)

	# plot vertical numerical results
	plt.plot(t, v_y_drift, c='k', label='analytics without history')
	plt.plot(t_history, v_y_drift_history, c='k', ls='--',
			 label='analytics with history')
	plt.scatter(t, w_d, edgecolors='k', facecolors='none', marker='o',
				label='numerics without history')
	plt.scatter(t_history, w_d_history, c='k', marker='x',
				label='numerics with history')
	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
