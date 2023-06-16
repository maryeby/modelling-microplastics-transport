import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

def main():
	"""
	This program plots the numerical and asymptotic results for a relaxing
	particle in a quiescent flow.
	"""
	# read data
	numerics = pd.read_csv('data/relaxing_numerics.csv')
	asymptotics = pd.read_csv('data/relaxing_asymptotics.csv')

	# generate plots
	plt.figure()
	plt.title('Relaxing Particle Velocity with History Effects', fontsize=18)
	plt.xlabel(r'$t$', fontsize=16)
	plt.ylabel(r'$\dot{x}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
#	plt.minorticks_on()
	plt.yscale('log')
	plt.ylim((1e-5, 1e1))

	plt.plot('t', 'light_xdot_history', c='hotpink', data=numerics,
			 label=r'light $\beta$')
	plt.plot('t', 'light_xdot', c='hotpink', ls='--', data=numerics, label='')
	plt.plot('t', 'light', c='hotpink', ls=':', data=asymptotics, label='')
	plt.plot('t', 'neutral_xdot_history', c='mediumpurple', data=numerics,
			 label=r'neutral $\beta$')
	plt.plot('t', 'neutral_xdot', c='mediumpurple', ls='--', data=numerics,
			 label='')
	plt.plot('t', 'neutral', c='mediumpurple', ls=':', data=asymptotics,
			 label='')
	plt.plot('t', 'heavy_xdot_history', c='cornflowerblue', data=numerics,
			 label=r'heavy $\beta$')
	plt.plot('t', 'heavy_xdot', c='cornflowerblue', ls='--', data=numerics,
			 label='')
	plt.plot('t', 'heavy', c='cornflowerblue', ls=':', data=asymptotics,
			 label='')
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
