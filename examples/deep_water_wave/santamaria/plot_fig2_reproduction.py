import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

def main():
	"""
	This program reproduces Figure 2 from Santamaria et al. (2013).
	"""
	# read data files
	numerics = pd.read_csv('../../data/deep_water_wave/'
						   + 'santamaria_fig2_recreation.csv')
	analytics = pd.read_csv('../../data/deep_water_wave/'
							+ 'santamaria_analytics.csv')

	# plot results
	plt.figure(1)
	plt.suptitle(r'Drift Velocity Comparison with Varying $\Delta t$',
				 fontsize=18)
	plt.subplot(121)
	plt.xlabel(r'$ \omega t $', fontsize=16)
	plt.ylabel(r'$ u_d / U $', fontsize=16)
#	plt.axis([0, 80, 0, 0.15])
#	plt.xticks(ticks=range(0, 80, 10), fontsize=14)
#	plt.yticks([0, 0.05, 0.1], fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)

	plt.plot('t', 'u_d', c='k', data=analytics, label='analytics')
	plt.axhline(0, c='k', ls=':', label='settling velocity')
	plt.scatter('sm_t', 'sm_u_d', c='k', marker='x', data=numerics,
				label='Santamaria numerics')
	plt.scatter('fine_t', 'fine_u_d', edgecolors='k', facecolors='none',
				data=numerics, label=r'Daitche numerics ($\Delta t =$ 1e-3)')
	plt.scatter('medium_t', 'medium_u_d', marker='s', edgecolors='k',
				facecolors='none', data=numerics,
				label=r'Daitche numerics ($\Delta t =$ 5e-3)')
	plt.scatter('coarse_t', 'coarse_u_d', marker='^', edgecolors='k',
				facecolors='none', data=numerics,
				label=r'Daitche numerics ($\Delta t =$ 1e-2)')
	plt.legend(fontsize=14)

	plt.subplot(122)
	plt.xlabel(r'$ \omega t $', fontsize=16)
	plt.ylabel(r'$ w_d / U $', fontsize=16)
#	plt.axis([0, 80, -0.128, -0.1245])
#	plt.xticks(ticks=range(0, 80, 10), fontsize=14)
#	plt.yticks([-0.128, -0.127, -0.126, -0.125], fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)

	plt.plot('t', 'w_d', c='k', data=analytics, label='analytics')
	plt.axhline(analytics['settling_velocity'].iloc[0], c='k', ls=':',
				label='settling velocity')
	plt.scatter('sm_t', 'sm_w_d', marker='x', c='k', data=numerics,
				label='Santamaria numerics')
	plt.scatter('fine_t', 'fine_w_d', edgecolors='k', facecolors='none',
				data=numerics, label=r'Daitche numerics ($\Delta t =$ 1e-3)')
	plt.scatter('medium_t', 'medium_w_d', marker='s', edgecolors='k',
				facecolors='none', data=numerics,
				label=r'Daitche numerics ($\Delta t =$ 5e-3)')
	plt.scatter('coarse_t', 'coarse_w_d', marker='^', edgecolors='k',
				facecolors='none', data=numerics,
				label=r'Daitche numerics ($\Delta t =$ 1e-2)')
	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
