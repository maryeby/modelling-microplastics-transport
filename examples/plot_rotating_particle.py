import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

def main():
	"""
	This program plots the numerical and analytical results for a rotating
	particle in a flow.
	"""
	# read data
	numerics = pd.read_csv('data/rotating_numerics.csv')
	int_times = pd.read_csv('data/rotating_int_times.csv')
	analytics = pd.read_csv('data/rotating_analytics.csv')
	daitche = pd.read_csv('data/daitche_fig3a.csv')

	# generate plots
	plt.figure()
	plt.title('Particle Trajectory: Rigid Body Rotation',
			  fontsize=18)
	plt.xlabel(r'$x$', fontsize=16)
	plt.ylabel(r'$z$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()
	plt.axis([-2, 2.5, -2.5, 2])

#	plt.plot('x_ana', 'z_ana', c='k', data=analytics, label='exact')
	plt.plot('exact_x', 'exact_z', c='k', data=daitche, label='exact extracted')
	plt.plot('first_x', 'first_z', c='k', ls='--', data=numerics,
			 label='first order')
	plt.plot('second_x', 'second_z', c='k', ls='-.', data=numerics,
			 label='second order')
	plt.plot('third_x', 'third_z', c='k', ls=':', data=numerics,
			 label='third order')
	plt.scatter('first_x', 'first_z', c='k', marker='x', data=daitche,
				label='1st order extracted')
	plt.scatter('int_x', 'int_z', c='k', marker='o', data=int_times,
				label='integer time steps')
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
