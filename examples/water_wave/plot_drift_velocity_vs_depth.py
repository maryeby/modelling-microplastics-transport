import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
	"""
	This program plots numerical and analytical solutions for the horizontal
	Stokes drift velocity of an inertial particle in linear water waves for
	varying Stokes numbers.
	"""
	# read data files
	numerics = pd.read_csv('../data/water_wave/drift_velocity_varying_st.csv')
	analytics = pd.read_csv('../data/water_wave/'
							+ 'neutrally_buoyant_analytics.csv')

	# create lists of labels and markers for scatter plots (numerical results)
	stokes_nums = numerics['St'][:].dropna()
	labels_1, labels_2 = ['deep', 'intermediate', 'shallow'], []
	markers = ['o', '^', 's', 'd']
	for St in stokes_nums:
		labels_2.append('%.2e' % St)

	# initialize figure
	plt.figure()
	plt.title(r'Stokes Drift Velocity Comparison with Varying St', fontsize=18)
	plt.xlabel(r'$\frac{u_d}{U\mathrm{Fr}}$', fontsize=16)
	plt.ylabel(r'$\frac{kz}{kh}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)

	# plot analytical solutions
	plt.plot('deep_u_d', 'deep_z/h', c='k', data=analytics,
			 label='deep analytics')
	plt.plot('intermediate_u_d', 'intermediate_z/h', '-.k', data=analytics,
			 label='intermediate analytics')
	plt.plot('shallow_u_d', 'shallow_z/h', ':k', data=analytics,
			 label='shallow analytics')

	# plot numerical solutions for each Stokes number
	for i in range(len(labels_1)):
		for j in range(len(stokes_nums)):
			if i == 0:
				label_3 = 'St = %g without history' % stokes_nums[j]
				label_4 = 'St = %g with history' % stokes_nums[j]
			else:
				label_3, label_4 = '', ''
			# numerics without history
			plt.scatter('%s_u_d_%g' % (labels_1[i], stokes_nums[j]),
						'%s_z/h' % labels_1[i], c='k', marker=markers[j],
						data=numerics, label=label_3)
			# numerics with history
			plt.scatter('%s_u_d_history_%g' % (labels_1[i], stokes_nums[j]),
						'%s_z/h' % labels_1[i], marker=markers[j],
						edgecolors='k', facecolors='none', data=numerics,
						label=label_4)
	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
