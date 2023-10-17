import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
	"""
	This program plots numerical and analytical solutions for the horizontal
	Stokes drift velocity of an inertial particle in linear deep water waves for
	varying Stokes numbers.
	"""
	# read data files
	numerics = pd.read_csv('../data/deep_water_wave/'
						   + 'drift_velocity_varying_st.csv')
	analytics = pd.read_csv('../data/deep_water_wave/'
							+ 'analytical_drift_velocity.csv')

	# create lists of labels and markers for scatter plots (numerical results)
	stokes_nums = numerics['St'][:].dropna()
	markers = ['o', '^', 's', 'd']

	# plot results
	plt.figure()
	plt.title(r'Stokes Drift Velocity Comparison with Varying St', fontsize=18)
	plt.xlabel(r'$u_d$', fontsize=16)
	plt.ylabel(r'$\frac{z}{h}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)

	plt.plot('u_d', 'z/h', c='k', data=analytics, label='analytics')
	for i in range(len(stokes_nums)):
		plt.scatter('u_d_%g' % stokes_nums[i], 'z/h', c='k', marker=markers[i],
					data=numerics, label='numerics (St = %g)' % stokes_nums[i])
	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
