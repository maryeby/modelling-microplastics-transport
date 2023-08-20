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
	This program computes analytical solutions for the Maxey-Riley equation
	without history for deep water waves, following the approach outlined in
	Santamaria et al. (2013), and saves the results to the `data` directory.
	"""
	# initialize the particle, flow, and transport system
	beta = 0.9
	my_particle = prt.Particle(stokes_num=0.157)
	my_flow = fl.DimensionalDeepWaterWave(amplitude=0.02, wavelength=1)
	my_system = ts.SantamariaTransportSystem(my_particle, my_flow, beta)

	# initialize variables for computing the analytics
	omega = my_flow.angular_freq
	t = np.arange(0, 80 / omega, 1e-3)
	z_0 = 0
	U = my_flow.max_velocity
	St = my_particle.stokes_num
	c = my_flow.phase_velocity
	bprime = 1 - my_system.density_ratio
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
	analytics.to_csv('../data/deep_water_wave/santamaria_analytics.csv',
					 index=False)

if __name__ == '__main__':
	main()
