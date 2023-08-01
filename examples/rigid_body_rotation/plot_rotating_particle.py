import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

def main():
	"""
	This program plots the numerical solutions, analyticsl solutions, and error
	analysis for a rotating particle in a flow.
	"""
	# read data
	numerics = pd.read_csv('../data/rotating_numerics.csv')
	analytics = pd.read_csv('../data/rotating_analytics.csv')
	int_times = pd.read_csv('../data/rotating_int_times.csv')
	rel_error = pd.read_csv('../data/rotating_rel_error.csv')
	global_error = pd.read_csv('../data/rotating_global_error.csv')
	computation_times = pd.read_csv('../data/rotating_computation_times.csv')
	daitche = pd.read_csv('../data/daitche_fig3a.csv')

#	trajectory_numerics = numerics
#	trajectory_analytics = analytics
#	trajectory_int_times = int_times
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
#	plt.axis([0, 100, 1e-7, 1e0])

	plt.plot('t', 'e_rel1', 'o--k', mfc='none', data=rel_error,
			 label='first order')
	plt.plot('t', 'e_rel2', 'x-.k', data=rel_error,
			 label='second order')
	plt.plot('t', 'e_rel3', '^:k', mfc='none', data=rel_error,
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
#	plt.axis([1e-5, 1e-1, 1e-11, 1e3])
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
