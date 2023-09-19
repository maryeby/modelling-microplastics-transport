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
	data_path = '../data/rigid_body_rotation/'
	numerics = pd.read_csv(data_path + 'numerics.csv')
	analytics = pd.read_csv(data_path + 'analytics.csv')
	int_times = pd.read_csv(data_path + 'int_times.csv')
	rel_error = pd.read_csv(data_path + 'rel_error.csv')
	global_error = pd.read_csv(data_path + 'global_error.csv')
	computation_times = pd.read_csv(data_path + 'computation_times.csv')
	daitche = pd.read_csv(data_path + 'daitche_fig3.csv')

	trajectory_numerics = numerics.iloc[:2000]
	trajectory_analytics = analytics.iloc[:2000]
	trajectory_int_times = int_times.iloc[:21]

	# generate plots
	plt.figure(1)
	plt.title('Particle Trajectory: Rigid Body Rotation',
			  fontsize=18)
	plt.xlabel(r'$x$', fontsize=16)
	plt.ylabel(r'$z$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()
	plt.axis([-2, 2.5, -2.5, 2])

	plt.plot('first_x', 'first_z', c='k', ls='--', data=trajectory_numerics,
			 label='first order')
	plt.plot('second_x', 'second_z', c='k', ls='-.', data=trajectory_numerics,
			 label='second order')
	plt.plot('third_x', 'third_z', c='k', ls=':', data=trajectory_numerics,
			 label='third order')
	plt.plot('x_ana', 'z_ana', c='k', data=trajectory_analytics, label='exact')
	plt.scatter('first_x', 'first_z', c='k', marker='x', data=daitche,
				label='1st order extracted')
	plt.scatter('int_x', 'int_z', c='k', marker='o', data=trajectory_int_times,
				label='integer time steps')
	plt.legend(fontsize=14)
	plt.tight_layout()

	plt.figure(2)
	plt.title(r'Relative Error with $\Delta t =$1e-2: Rigid Body Rotation',
			  fontsize=18)
	plt.xlabel(r'$t$', fontsize=16)
	plt.ylabel(r'$E_{rel}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.yscale('log')
	plt.minorticks_on()
	plt.axis([0, 100, 1e-7, 1e0])

	plt.plot('t1', 'rel_error1', c='grey', data=daitche, label='')
	plt.plot('t2', 'rel_error2', c='grey', data=daitche, label='')
	plt.plot('t3', 'rel_error3', c='grey', data=daitche, label='')
	plt.plot('t', 'e_rel1', '--k', mfc='none', data=rel_error,
			 label='first order')
	plt.plot('t', 'e_rel2', '-.k', data=rel_error,
			 label='second order')
	plt.plot('t', 'e_rel3', ':k', mfc='none', data=rel_error,
			 label='third order')
	plt.legend(fontsize=14)
	plt.tight_layout()

	plt.figure(3)
	plt.title('Global Error: Rigid Body Rotation',
			  fontsize=18)
	plt.xlabel(r'$\Delta t$', fontsize=16)
	plt.ylabel(r'$\mathcal{\epsilon}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.minorticks_on()
	plt.axis([1e-3, 1e-1, 1e-11, 1e3])
	h_scale = np.linspace(2e-3, 5e-2, 10)

	plt.plot(h_scale, h_scale, c='grey', ls='--', label=r'~$h$')
	plt.plot(h_scale, h_scale ** 2, c='grey', ls='-.', label=r'~$h^2$')
	plt.plot(h_scale, h_scale ** 3, c='grey', ls=':', label='~$h^3$')
	plt.plot('delta_t', 'global_error1', '.--k', data=global_error,
			 label='first order')
	plt.plot('delta_t', 'global_error2', '.-.k', data=global_error,
			 label='second order')
	plt.plot('delta_t', 'global_error3', '.:k', data=global_error,
			 label='third order')
	plt.legend(fontsize=14)
	plt.tight_layout()

	plt.figure(4)
	plt.title('Timestep Size vs Computation Time: Rigid Body Rotation',
			  fontsize=18)
	plt.xlabel(r'$\Delta t$', fontsize=16)
	plt.ylabel('computation time (s)', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.minorticks_on()
	plt.plot('delta_t', 'computation_time', '.-k', data=computation_times)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
