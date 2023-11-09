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
	stokes_nums = numerics['St'][:].drop_duplicates()
	markers = ['o', '^', 's', 'd']

	# plot results
	plt.figure()
	plt.title(r'Deep Water Stokes Drift Velocity with Varying St', fontsize=18)
	plt.xlabel(r'$\frac{u_d}{U\mathrm{Fr}}$', fontsize=16)
	plt.ylabel(r'$\frac{kz}{kh}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	plt.plot('u_d', 'z/h', c='k', data=analytics, label='analytics')
	m = 0
	for St in stokes_nums:
		u_d = numerics['u_d'].where(numerics['St'] == St).dropna()
		z = numerics['z_0'].where(numerics['St'] == St).dropna()
		h = numerics['h'].where(numerics['St'] == St).dropna()
		plt.scatter(u_d, z/h, c='k', marker=markers[m],
					label='numerics (St = %g)' % St)
		m += 1
	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
