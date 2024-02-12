import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import scipy.constants as constants

from transport_framework import particle as prt
from models import dim_deep_water_wave as fl
from models import santamaria_system as ts

def main():
	"""
	This program computes analytical solutions for the Stokes drift velocity of
	an inertial particle in linear water waves, following the approach outlined
	in Santamaria et al. (2013), and saves the results to the
	`data/deep_water_wave` directory.
	"""
	# initialize the flow and related parameters
	my_flow = fl.DimensionalDeepWaterWave(amplitude=0.02, wavelength=1)
	omega = my_flow.angular_freq
	U = my_flow.max_velocity
	c = my_flow.phase_velocity

	# initialize the remaining variables for computing the analytics
	num_periods = 50
	delta_t = 1e-3
	t = np.arange(0, num_periods * my_flow.period, delta_t)
	z_0 = 0
	beta = 0.9
	bprime = 1 - beta
	St = 0.157
	e_2z0t = np.exp(2 * (my_flow.wavenum * z_0 - St * bprime * t * omega))

	# compute analytical drift velocity and settling velocity
	u_d = U ** 2 / c * e_2z0t * (1 - St ** 2 * bprime)
	w_d = -c * St * bprime * (1 + 2 * (U / c) ** 2 * e_2z0t)
	settling_velocity = -bprime * constants.g * (St / omega)

	# store results and write to csv file
	my_dict = {'t': t * omega, 'u_d': u_d / U, 'w_d': w_d / U,
			   'settling_velocity': settling_velocity / U}
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	analytics = pd.DataFrame(my_dict)
	analytics.to_csv('../../data/deep_water_wave/santamaria_analytics.csv',
					 index=False)

if __name__ == '__main__':
	main()
