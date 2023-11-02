import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import scipy.constants as constants
from models import dim_water_wave as fl

def main():
	r"""
	This program computes analytical solutions for the horiztonal Stokes drift
	velocity of an inertial particle in a linear water wave and saves the
	results to the `data/water_wave` directory.
	"""
	# read numerical data
	numerics = pd.read_csv('../data/water_wave/drift_velocity_varying_st.csv')	

	# compute asymptotics
	my_dict = {}
	compute_drift_velocity('deep', my_dict, numerics)
	compute_drift_velocity('intermediate', my_dict, numerics)
	compute_drift_velocity('shallow', my_dict, numerics)

	# store results and write to csv files
	analytics = pd.DataFrame(my_dict)
	analytics.to_csv('../data/water_wave/neutrally_buoyant_analytics.csv',
					 index=False)

def compute_drift_velocity(label, my_dict, numerics):
	r"""
	Computes the horizontal Stokes drift velocity using the following
	equation from Van den Bremer and Breivik (2017),
	$$u_d = c(Ak)^2 \frac{\cosh{(2k(z + h))}}{2\sinh^2(kh)}.$$

	Parameters
	----------
	label : str
		The label used to access the initial depth values.
	my_dict : dictionary
		The dictionary to store the results.
	numerics : DataFrame
		The numerical results used to access the initial depth values.

	Returns
	-------
	u_d : float
		The normalized horizontal Stokes drift velocity.
	"""
	# initialize the flow (wave)
	h = numerics[label + '_h'].iloc[0]
	A = numerics['amplitude'].iloc[0]
	wavelength = numerics['wavelength'].iloc[0]
	my_wave = fl.DimensionalWaterWave(depth=h * wavelength / (2 * np.pi),
									  amplitude=A, wavelength=wavelength)

	# initialize parameters for computations
	k = my_wave.wavenum
	Fr = my_wave.froude_num
	U = my_wave.max_velocity
	z = np.linspace(numerics[label + '_z'].iloc[0],
					numerics[label + '_z'].iloc[-1], 100)

	# dimensionalize z and h, compute drift velocity
	z /= k
	h /= k
	u_d = U * Fr * np.cosh(2 * k * (z + h)) / (2 * np.sinh(k * h) ** 2)

	# store normalized results
	my_dict[label + '_u_d'] = u_d / (U * Fr)
	my_dict[label + '_z/h'] = k * z / (k * h)

if __name__ == '__main__':
	main()
