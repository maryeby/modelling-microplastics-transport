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
	numerics = pd.read_csv('../data/deep_water_wave/'
						   + 'santamaria_fig2_recreation.csv')
	analytics = pd.read_csv('../data/deep_water_wave/santamaria_analytics.csv')

	# plot results
	plt.figure(1)
	plt.title(r'Drift Velocity Comparison with Varying $\Delta t$', fontsize=18)
	plt.subplot(121)
	plt.xlabel(r'$ \omega t $', fontsize=16)
	plt.ylabel(r'$ u_d / U $', fontsize=16)
	plt.axis([0, 80, 0, 0.15])
	plt.xticks(ticks=range(0, 80, 10), fontsize=14)
	plt.yticks([0, 0.05, 0.1], fontsize=14)

	plt.plot('t', 'u_d', c='k', data=analytics, label='analytical')
	plt.scatter('fine_t', 'fine_u_d', edgecolors='k', facecolors='none',
				data=numerics, label=r'numerical $\Delta t =$ 1e-3')
#	plt.scatter('t', 'medium_u_d', marker='s', edgecolors='k',
#				facecolors='none', data=numerics,
#				label=r'numerical $\Delta t =$ 5e-3')
#	plt.scatter('t', 'coarse_u_d', edgecolors='k', marker='x', label=r'numerical $\Delta t =$ 1e-2')
	plt.legend(fontsize=14)

	plt.subplot(122)
	plt.xlabel(r'$ \omega t $', fontsize=16)
	plt.ylabel(r'$ w_d / U $', fontsize=16)
	plt.axis([0, 80, -0.128, -0.1245])
	plt.xticks(ticks=range(0, 80, 10), fontsize=14)
	plt.yticks([-0.128, -0.127, -0.126, -0.125], fontsize=14)

	plt.plot('t', 'w_d', c='k', data=analytics, label='analytical')
	plt.axhline(analytics['settling_velocity'].iloc[0], c='k', ls=':',
				label='settling velocity')
	plt.scatter('fine_t', 'fine_w_d', edgecolors='k', facecolors='none',
				data=numerics, label=r'numerical $\Delta t =$ 1e-3')
#	plt.scatter('t', 'medium_w_d', marker='s', edgecolors='k',
#				facecolors='none', data=numerics,
#				label=r'numerical $\Delta t =$ 5e-3')
#	plt.scatter('t', 'coarse_w_d', marker='x',
#				data=numerics, label=r'numerical $\Delta t =$ 1e-2')
	plt.legend(fontsize=14)
	plt.tight_layout()

	plt.figure(2)
	plt.title('Particle Trajectory', fontsize=18)
	plt.xlabel('x', fontsize=16)
	plt.ylabel('z', fontsize=16)
	plt.plot('x', 'z', '.k-', zorder=1, data=numerics, label='my numerics')
	plt.scatter('interpd_x', 'interpd_z', zorder=2, c='coral', data=numerics,
				label='endpoints')
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
