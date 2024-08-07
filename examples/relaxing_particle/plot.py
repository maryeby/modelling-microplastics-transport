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
	numerics = pd.read_csv('../data/relaxing_particle/numerics.csv')
	asymptotics = pd.read_csv('../data/relaxing_particle/asymptotics.csv')
	prasath = pd.read_csv('../data/relaxing_particle/prasath_fig4.csv')

	# generate plots
	plt.figure()
	fs, lfs = 14, 16
	colors = ['#d7301f', '#fc8d59', '#fdcc8a']
#	plt.title('Relaxing Particle Velocity with Third Order History Effects',
#			  fontsize=18)
	plt.xlabel(r'$t$', fontsize=lfs)
	plt.ylabel(r'$\dot{x}$', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()
	plt.yscale('log')
	plt.axis([0, 14.5, 1e-5, 1e1])

	plt.plot('t', 'light_xdot_history', c=colors[0], data=numerics,
			 label=r'$\beta = 0.01$')
	plt.plot('t', 'light_xdot', c=colors[0], ls='--', data=numerics, label='')
	plt.plot('t', 'light', c=colors[0], ls=':', data=asymptotics, label='')
	plt.plot('t', 'neutral_xdot_history', c=colors[1], data=numerics,
			 label=r'$\beta = 1$')
	plt.plot('t', 'neutral_xdot', c=colors[1], ls='--', data=numerics,
			 label='')
	plt.plot('t', 'neutral', c=colors[1], ls=':', data=asymptotics, label='')
	plt.plot('t', 'heavy_xdot_history', c=colors[2], data=numerics,
			 label=r'$\beta = 5$')
	plt.plot('t', 'heavy_xdot', c=colors[2], ls='--', data=numerics, label='')
	plt.plot('t', 'heavy', c=colors[2], ls=':', data=asymptotics, label='')
#	plt.plot('light_t_history', 'light_xdot_history', c='silver', data=prasath,
#			 label='')
#	plt.plot('light_t', 'light_xdot', c='silver', ls='--', data=prasath,
#			 label='')
#	plt.plot('light_t_asymp', 'light_xdot_asymp', c='silver', ls=':',
#			 data=prasath, label='')
#	plt.plot('neutral_t_history', 'neutral_xdot_history', c='silver',
#			 data=prasath, label='')
#	plt.plot('neutral_t', 'neutral_xdot', c='silver', ls='--', data=prasath,
#			 label='')
#	plt.plot('neutral_t_asymp', 'neutral_xdot_asymp', c='silver', ls=':',
#			 data=prasath, label='')
#	plt.plot('heavy_t_history', 'heavy_xdot_history', c='silver', data=prasath,
#			 label='')
#	plt.plot('heavy_t', 'heavy_xdot', c='silver', ls='--', data=prasath,
#			 label='')
#	plt.plot('heavy_t_asymp', 'heavy_xdot_asymp', c='silver', ls=':',
#			 data=prasath, label='')
	plt.legend(fontsize=fs)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
