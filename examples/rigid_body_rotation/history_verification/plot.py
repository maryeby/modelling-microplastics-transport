import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
	"""
	This program plots the numerical solutions, analyticsl solutions, and error
	analysis for a rotating particle in a flow.
	"""
	# read data
	data_path = '../../data/rigid_body_rotation/'
	analytics = pd.read_csv(data_path + 'analytical_history.csv')
	numerics = pd.read_csv(data_path + 'numerical_history.csv')
	rel_error = pd.read_csv(data_path + 'rel_error_history.csv')
#	global_error = pd.read_csv(data_path + 'global_error.csv')
	history_x_num, history_z_num = numerics['H\'_x'], numerics['H\'_z']
	history_x, history_z = analytics['H\'_x'], analytics['H\'_z']
	F_x, F_z = analytics['F_x'], analytics['F_z']
	x_num, z_num, t_num = numerics['x'], numerics['z'], numerics['t']
	x, z, t = analytics['x'], analytics['z'], analytics['t']
	n = int(t.size - 2)
	history_x_num[0] = 0

	# generate plots
	fs, lfs = 14, 16
	plt.figure(1)
	plt.subplot(211) # horizontal component
	plt.ylabel(r'$H\'(t)_x$', fontsize=lfs)
	plt.xticks([])
	plt.yticks(fontsize=fs)
	plt.minorticks_on()
	plt.plot(t, F_x, '--k', label='Candelier analytics')
	plt.plot(t[:n], history_x[:n], c='silver', label='Daitche analytics')
	plt.plot(t_num[:n], history_x_num[:n], ':k', label='numerics')
	plt.legend(fontsize=fs)

	plt.subplot(212) # vertical component
	plt.xlabel(r'$t$', fontsize=lfs)
	plt.ylabel(r'$H\'(t)_z$', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()
	plt.plot(t, F_z, '--k')
	plt.plot(t[:n], history_z[:n], c='silver')
	plt.plot(t_num[:n], history_z_num[:n], ':k')
	plt.tight_layout()
	
	plt.figure(2)
	plt.xlabel('x', fontsize=lfs)
	plt.ylabel('z', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()
	plt.gca().set_box_aspect(1)
	plt.plot('x', 'z', c='silver', data=analytics, label='Candelier analytics')
	plt.plot('x', 'z', ':k', data=numerics, label='numerics')
	plt.legend(fontsize=fs)

	plt.figure(3)
	plt.xlabel('t', fontsize=lfs)
	plt.ylabel(r'$E_{abs}$', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.yscale('log')
	plt.axis([-0.5, 10.5, 1e-8, 1e-6])
	plt.minorticks_on()
	plt.plot('t', 'E_abs', '-k', data=rel_error)
	plt.show()

if __name__ == '__main__':
	main()
