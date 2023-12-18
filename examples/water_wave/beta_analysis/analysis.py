import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd
import itertools
import scipy.constants as constants
from scipy.optimize import curve_fit

from models import my_system as ts

DATA_PATH = '../../data/water_wave/'

def main():
	r"""
	This program computes the Stokes drift velocity for inertial particles of
	varying buoyancies in linear water waves and saves the results to the
	`data/water_wave` directory.
	"""
	# initialize variables
	betas = [0.5, 0.9]
	z_0s = np.linspace(-0.25, -4, 10, endpoint=False)
	St = 0.01
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	z_list, u_d_list, beta_list, history_list, exact_list = [], [], [], [], []

	# read data
	in_file = 'numerics.csv'
	out_file = 'beta_analysis.csv'
	numerics = pd.read_csv(DATA_PATH + in_file)

	# analysis for neutrally buoyant particles (beta = 1)
	for i in itertools.product(z_0s, [True, False]):
		z_0, history = i
		beta = 1

		# create condition to help filter through numerical data
		cond = (numerics['z_0'] == z_0) & (numerics['St'] == St) \
									  & (numerics['beta'] == beta) \
									  & (numerics['history'] == history) \
									  & (numerics['h'] == h) \
									  & (numerics['A'] == A) \
									  & (numerics['wavelength'] == wavelength) \
									  & (numerics['delta_t'] == 5e-4)
		# retrieve relevant data
		Fr = numerics['Fr'].where(cond).dropna().iloc[0]
		x = numerics['x'].where(cond).dropna().to_numpy()
		z = numerics['z'].where(cond).dropna().to_numpy()
		xdot = numerics['xdot'].where(cond).dropna().to_numpy()
		t = numerics['t'].where(cond).dropna().to_numpy()

		# compute and scale drift velocity
		_, _, u_d, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
		avg_u_d = np.average(u_d) / Fr

		# store solutions
		u_d_list.append(avg_u_d)
		z_list.append(z_0)
		beta_list.append(beta)
		history_list.append(history)
		exact_list.append(True)

	# analysis for negatively buoyant particles (beta < 1)
	for i in itertools.product(betas, [True, False]):
		beta, history = i

		# create condition to help filter through numerical data
		cond = (numerics['St'] == St) & (numerics['beta'] == beta) \
									  & (numerics['history'] == history) \
									  & (numerics['h'] == h) \
									  & (numerics['A'] == A) \
									  & (numerics['wavelength'] == wavelength) \
									  & (numerics['delta_t'] == delta_t)
		# retrieve relevant data
		U = numerics['U'].where(cond).dropna().iloc[0]
		Fr = numerics['Fr'].where(cond).dropna().iloc[0]
		k = numerics['k'].where(cond).dropna().iloc[0]
		x = numerics['x'].where(cond).dropna().to_numpy()
		z = numerics['z'].where(cond).dropna().to_numpy()
		xdot = numerics['xdot'].where(cond).dropna().to_numpy()
		t = numerics['t'].where(cond).dropna().to_numpy()

		# compute and scale drift velocity
		_, z_crossings, u_d, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
		u_d /= Fr

		# store exact solutions
		u_d_list += u_d.tolist()
		z_list += z_crossings[1:].tolist()
		beta_list += [beta] * len(u_d)
		history_list += [history] * len(u_d)
		exact_list += [True] * len(u_d)

		# fit an exponential curve to the data
		f = lambda x, a, b, c, d : a * np.exp(b * x) + c * np.exp(d * x)
		z = z_crossings[1:]
		coefficients, covariance = curve_fit(f, z, u_d)
		a, b, c, d = coefficients
		extended_range = np.linspace(0, z[-1], 100)
		u_d = f(extended_range, a, b, c, d)

		# store estimated solutions
		u_d_list += u_d.tolist()
		z_list += extended_range.tolist()
		beta_list += [beta] * len(u_d)
		history_list += [history] * len(u_d)
		exact_list += [False] * len(u_d)

	# compute analytical solutions
	analytical_z = np.linspace(0, -5, 100) / k
	analytical_u_d = U * Fr * np.cosh(2 * k * (analytical_z + h)) \
					   / (2 * np.sinh(k * h) ** 2)
	analytical_z *= k
	analytical_u_d /= (U * Fr)

	# write results to data file
	results_dict = {'analytical_z': analytical_z,
					'analytical_u_d': analytical_u_d, 'z': z_list,
					'u_d': u_d_list, 'beta': beta_list, 'history': history_list,
					'exact': exact_list}
	results_dict = dict([(key, pd.Series(value)) for key, value in
						  results_dict.items()])
	results = pd.DataFrame(results_dict)
	results.to_csv(DATA_PATH + out_file, index=False)

if __name__ == '__main__':
	main()
