import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '../../data/water_wave/'

def main():
	"""
	This program plots numerical and analytical solutions for the horizontal
	Stokes drift velocities of inertial particles in linear water waves for
	varying Stokes numbers and water depths.
	"""
	# read data files
	numerics = pd.read_csv(DATA_PATH + 'drift_velocity_varying_st.csv')
	analytics = pd.read_csv(DATA_PATH + 'neutrally_buoyant_analytics.csv')

	# create lists of labels and markers for scatter plots (numerical results)
	stokes_nums = numerics['St'][:].dropna()
	labels = ['deep', 'intermediate', 'shallow']
	markers = ['o', '^', 's', 'd']

	# initialize figure
	plt.figure()
	plt.title(r'Stokes Drift Velocity Comparison with Varying St', fontsize=18)
	plt.xlabel(r'$\frac{u_d}{U\mathrm{Fr}}$', fontsize=16)
	plt.ylabel(r'$\frac{kz}{kh}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# plot analytical solutions
	plt.plot('deep_u_d', 'deep_z/h', c='k', data=analytics,
			 label='deep')
	plt.plot('intermediate_u_d', 'intermediate_z/h', '-.k', data=analytics,
			 label='intermediate')
	plt.plot('shallow_u_d', 'shallow_z/h', ':k', data=analytics,
			 label='shallow')

	# plot numerical solutions for each Stokes number
	for i in range(len(labels)):
		for j in range(len(stokes_nums)):
			if i == 0:
#				no_history_label = 'St = %g without history' % stokes_nums[j]
				no_history_label = 'St = %g' % stokes_nums[j]
				history_label = 'St = %g with history' % stokes_nums[j]
			else:
				no_history_label, history_label = '', ''
			# numerics without history
			plt.scatter('%s_u_d_%g' % (labels[i], stokes_nums[j]),
						'%s_z/h' % labels[i], c='k', marker=markers[j],
						data=numerics, label=no_history_label)
			# numerics with history
			plt.scatter('%s_u_d_history_%g' % (labels[i], stokes_nums[j]),
						'%s_z/h' % labels[i], marker=markers[j],
						edgecolors='k', facecolors='none', data=numerics,
						label='')
#						label=history_label)
	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
