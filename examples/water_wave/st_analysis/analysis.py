import sys
import warnings
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import itertools
import scipy.constants as constants
from scipy.optimize import curve_fit

from models import my_system as ts

DATA_PATH = '../../data/water_wave/'

def main():
	r"""
	This program computes the Stokes drift velocity for inertial particles with
	varying Stokes numbers in linear water waves and saves the results to the
	`data/water_wave` directory.
	"""
	# initialize variables
	x_0, z_0 = 0, 0
	beta = 0.9
	stokes_nums = [0.01, 0.1, 1]
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	z_list, u_d_list, St_list, history_list, exact_list = [], [], [], [], []

	# read data
	in_file = 'numerics.csv'
	out_file = 'st_analysis.csv'
	numerics = pd.read_csv(DATA_PATH + in_file)

	# analysis for negatively buoyant particles (beta < 1)
	for i in itertools.product(stokes_nums, [True, False]):
		St, history = i

		# create condition and lambdas to help filter through numerical data
		condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
									& (numerics['St'] == St) \
									& (numerics['beta'] == beta) \
									& (numerics['history'] == history) \
									& (numerics['h\''] == h) \
									& (numerics['A\''] == A) \
									& (numerics['wavelength\''] == wavelength) \
									& (numerics['delta_t\''] == delta_t)
		get_single = lambda name : numerics[name].where(condition).dropna()\
												 .iloc[0]
		get_series = lambda name : numerics[name].where(condition).dropna()\
												  .to_numpy()
		# retrieve relevant data
		k = get_single('k\'')
		A = get_single('A\'')
		x = get_series('x')
		z = get_series('z')
		xdot = get_series('xdot')
		t = get_series('t')

		# compute and scale drift velocity
		_, z_crossings, u_d, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
		u_d /= k * A

		# store exact solutions
		u_d_list += u_d.tolist()
		z_list += z_crossings[1:].tolist()
		St_list += [St] * len(u_d)
		history_list += [history] * len(u_d)
		exact_list += [True] * len(u_d)

		# perform curve fitting if there are numerical solutions
		z = z_crossings[1:]
		if np.any(z):
			extended_range = np.linspace(0, z[-1] - 1, 100)

			if St == 1:
				f = lambda x, a, b : a * np.exp(b / x)
			else:
				f = lambda x, a, b, c, d, offset : a * np.exp(b * x) \
												 + c * np.exp(d * x) + offset
			# fit curve to data
			coefficients, covariance = curve_fit(f, z, u_d)
			if St == 1:
				a, b = coefficients
				u_d = f(extended_range, a, b)
			else:
				a, b, c, d, offset = coefficients
				u_d = f(extended_range, a, b, c, d, offset)

			# store estimated solutions
			u_d_list += u_d.tolist()
			z_list += extended_range.tolist()
			St_list += [St] * len(u_d)
			history_list += [history] * len(u_d)
			exact_list += [False] * len(u_d)

	# write results to data file
	results_dict = {'z': z_list, 'u_d': u_d_list, 'St': St_list,
					'history': history_list, 'exact': exact_list}
	results_dict = dict([(key, pd.Series(value)) for key, value in
						  results_dict.items()])
	results = pd.DataFrame(results_dict)
	results.to_csv(DATA_PATH + out_file, index=False)

if __name__ == '__main__':
	main()
