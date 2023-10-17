import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import scipy.constants as constants
from models import deep_water_wave as fl

def main():
	r"""
	This program computes analytical solutions for the horizontal Stokes drift
	velocity of an inertial particle in linear deep water waves and saves the
	results to the `data/deep_water_wave` directory.
	"""
	# read numerical data
	numerics = pd.read_csv('../data/deep_water_wave/'
						   + 'drift_velocity_varying_st.csv')
	h = numerics['h'].iloc[0]
	A = numerics['A'].iloc[0]
	wavelength = numerics['lambda'].iloc[0]
	z = np.linspace(numerics['z'].iloc[0], numerics['z'].iloc[-1], 100)

	# initialize the flow (wave) and related parameters
	my_wave = fl.DeepWaterWave(depth=h, amplitude=A, wavelength=wavelength)
	c = my_wave.phase_velocity
	Fr = my_wave.froude_num
	U = my_wave.max_velocity
	k = my_wave.wavenum

	# compute drift velocity
	z /= k # dimensional z
	u_d = c * Fr ** 2 * np.exp(2 * k * z)

	# normalize and store results
	z *= k
	my_dict = {}
	my_dict['u_d'] = u_d / (U * Fr)
	my_dict['z/h'] = z / h

	# store results and write to csv files
	analytics = pd.DataFrame(my_dict)
	analytics.to_csv('../data/deep_water_wave/analytical_drift_velocity.csv',
					 index=False)

if __name__ == '__main__':
	main()
