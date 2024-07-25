import sys 
import warnings
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit

from models import my_system as ts
DATA_PATH = '../../data/water_wave/'

def main():
	"""
	This program computes numerical solutions for the Stokes drift velocity of
	a negatively buoyant inertial particle in a linear water wave and saves the
	results to the `data/water_wave` directory.
	"""
	x_0, z_0 = 0, 0
	# initialize variables
	St, beta = 0.01, 0.8
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	R = 2 / 3 * beta

	# read numerical data
	in_file = 'numerics.csv'
	out_file = 'velocity_analysis.csv'
	numerics = pd.read_csv(DATA_PATH + in_file)

	# create condition and lambdas to help filter through numerical data
	condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
									& (numerics['St'] == St) \
									& (numerics['beta'] == beta) \
									& (numerics['h\''] == h) \
									& (numerics['A\''] == A) \
									& (numerics['wavelength\''] == wavelength) \
									& (numerics['delta_t\''] == delta_t)
	get = lambda name, history : numerics[name].where(condition \
											& (numerics['history'] == history))\
													  .dropna().to_numpy()
	# retrieve relevant numerical results
	x = get('x', False)
	z = get('z', False)
	xdot = get('xdot', False)
	t = get('t', False)

	x_history = get('x', True)
	z_history = get('z', True)
	xdot_history = get('xdot', True)
	t_history = get('t', True)

	# compute drift velocity
	x_crossings, z_crossings, u_d, w_d, t_d = ts.compute_drift_velocity(x, z,
																		xdot, t)
	x_crossings_history, z_crossings_history, \
		u_d_history, w_d_history, t_d_history = ts.compute_drift_velocity(
												   x_history, z_history,
												   xdot_history, t_history)

	# fit curve to horizontal drift velocity without history data
	f = lambda x, a, b, c, d : a * np.exp(b * x) + c * np.exp(d * x)
	coefficients, covariance = curve_fit(f, u_d, t_d)
	a, b, c, d = coefficients
	fitted_t_u = f(u_d, a, b, c, d)

	# fit curve to horizontal drift velocity with history data
	coefficients, covariance = curve_fit(f, u_d_history, t_d_history)
	a, b, c, d = coefficients
	fitted_t_u_history = f(u_d_history, a, b, c, d)

	# fit curve to vertical drift velocity without history data
	f = lambda x, a, b, c, d, offset : np.exp((a + b * x) / (c + d * x)) \
												 + offset
	coefficients, covariance = curve_fit(f, w_d, t_d)
	a, b, c, d, offset = coefficients
	fitted_t_w = f(w_d, a, b, c, d, offset)

	# fit curve to horizontal drift velocity with history data
	f = lambda x, a, b, c, offset : a ** (np.log(b * x + c)) + offset
	coefficients, covariance = curve_fit(f, w_d_history, t_d_history)
	a, b, c, offset = coefficients
	fitted_t_w_history = f(w_d_history, a, b, c, offset)

	# write results to data file
	history = [True] * len(x_crossings)
	no_history = [False] * len(x_crossings)
	results = {'x_crossings': x_crossings, 'z_crossings': z_crossings,
			'u_d': u_d, 'w_d': w_d, 't': t_d, 'fitted_t_u': fitted_t_u,
			'fitted_t_w': fitted_t_w, 'history': no_history}
	results_history = {'x_crossings': x_crossings_history,
					   'z_crossings': z_crossings_history,
					   'u_d': u_d_history, 'w_d': w_d_history,
					   't': t_d_history, 'fitted_t_u': fitted_t_u_history,
					   'fitted_t_w': fitted_t_w_history, 'history': history}
	results = dict([(key, pd.Series(value)) for key, value in results.items()])
	results_history = dict([(key, pd.Series(value)) for key, value
							in results_history.items()])
	df1, df2 = pd.DataFrame(results), pd.DataFrame(results_history)
	df = pd.concat([df1, df2])
	df.to_csv(DATA_PATH + out_file, index=False)

if __name__ == '__main__':
	main()
