import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import scipy.constants as constants
from models import dim_deep_water_wave as fl

def main():
	"""
	This program computes analytical solutions for the Stokes drift velocity of
	an inertial particle in linear water waves, following the approach outlined
	in Santamaria et al. (2013), and saves the results to the
	`data/water_wave` directory.
	"""
	# read beta values from numerics
	numerics = pd.read_csv('../data/water_wave/drift_velocity_varying_beta.csv')
	betas = numerics['beta'][:].dropna()
	A = numerics['amplitude'].iloc[0]
	wavelength = numerics['wavelength'].iloc[0]

	# initialize the flow (wave) and related parameters
	my_wave = fl.DimensionalDeepWaterWave(amplitude=A, wavelength=wavelength)
	omega = my_wave.angular_freq
	U = my_wave.max_velocity
	Fr = my_wave.froude_num
	c = my_wave.phase_velocity

	# initialize the remaining parameters for analytical computations
	T = omega * Fr
	num_periods = 50
	delta_t = 5e-3
	t = np.arange(0, num_periods * my_wave.period, delta_t)
	z_0 = 0
	St = 0.157

	# create dictionary to store results
	my_dict = {}
	my_dict['t'] = t * omega

	# compute analytical solutions for each value of beta
	for beta in betas:
		bprime = 1 - beta
		e_2z0t = np.exp(2 * (my_wave.wavenum * z_0 - St * bprime * t * omega))

		# compute drift velocity and settling velocity
		u_d = U ** 2 / c * e_2z0t * (1 - St ** 2 * bprime)
		w_d = -c * St * bprime * (1 + 2 * (U / c) ** 2 * e_2z0t)
		settling_velocity = -bprime * constants.g * (St / omega)

		# store results in dictionary
		my_dict['u_d_%g' % beta] = u_d / (U * Fr)
		my_dict['w_d_%g' % beta] = w_d / (U * Fr)
		my_dict['settling_velocity_%g' % beta] = settling_velocity / (U * Fr)

	# write to csv file
	analytics = pd.DataFrame(my_dict)
	analytics.to_csv('../data/water_wave/santamaria_analytics.csv',
					 index=False)

if __name__ == '__main__':
	main()
