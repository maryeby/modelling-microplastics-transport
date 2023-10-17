import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
	"""
	This program plots numerical and analytical solutions for the Stokes drift
	velocity of inertial particles of varying densities in linear water waves.
	"""
	# read data
	numerics = pd.read_csv('../data/water_wave/drift_velocity_varying_beta.csv')
	analytics = pd.read_csv('../data/water_wave/santamaria_analytics.csv')

	# create lists of labels and markers
	betas = numerics['beta'][:].dropna()
	colors = ['k', 'coral', 'mediumpurple', 'cornflowerblue']

	# create a separate figure for each value of beta
	for i in range(len(betas)):
		# initialize drift velocity figure & left subplot
		plt.figure(i + 1)
		plt.suptitle(r'Time vs Stokes Drift for $\beta = $%g' % betas[i],
					 fontsize=18)
		plt.subplot(121)
		plt.xlabel(r'$\omega t$', fontsize=16)
		plt.ylabel(r'$\frac{u_d}{U}$', fontsize=16)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)

		# plot horizontal results
		plt.plot('t', 'u_d_%g' % betas[i], c=colors[0], ls='-.', data=analytics,
				 label='analytics')
		plt.plot('t_%g' % betas[i], 'u_d_%g' % betas[i],
				 c=colors[0], ls='--', marker='.', data=numerics,
				 label='numerics without history')
#		plt.plot('t_history_%g' % betas[i], 'u_d_history_%g' % betas[i],
#				 c=colors[0], marker='.', data=numerics,
#				 label='numerics with history')
		plt.legend(fontsize=14)

		# initialize right subplot
		plt.subplot(122)
		plt.xlabel(r'$\omega t$', fontsize=16)
		plt.ylabel(r'$\frac{w_d}{U}$', fontsize=16)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)

		# plot vertical results
		plt.axhline(analytics['settling_velocity_%g' %  betas[i]].iloc[0],
					c=colors[0], ls=':',
					label='settling velocity')
		plt.plot('t', 'w_d_%g' % betas[i], c=colors[0], ls='-.', data=analytics,
				 label='analytics')
		plt.plot('t_%g' % betas[i], 'w_d_%g' % betas[i],
				 c=colors[0], marker='.', ls='--', data=numerics,
				 label='numerics without history')
#		plt.plot('t_history_%g' % betas[i], 'w_d_history_%g' % betas[i],
#				 c=colors[0], marker='.', data=numerics,
#				 label='numerics with history')
		plt.legend(fontsize=14)

		# initialize trajectory figure
		plt.figure(i + len(betas) + 1)
		plt.title(r'Trajectory for $\beta = $%g' % betas[i], fontsize=18)
		plt.xlabel('x', fontsize=14)
		plt.ylabel('z', fontsize=14)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)

		# plot particle trajectory with and without history
		plt.plot('x_%g' % betas[i], 'z_%g' % betas[i], marker='.',
				 c=colors[0], ls='--', data=numerics, label='without history')
#		plt.plot('x_history_%g' % betas[i], 'z_history_%g' % betas[i],
#				 marker='.', c=colors[0], data=numerics, label='with history')
		plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
