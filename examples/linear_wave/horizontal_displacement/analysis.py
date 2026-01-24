import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.optimize import curve_fit

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, update_results
from utils.colors import print_failure
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 DEPTH, WAVELENGTH
from examples.linear_wave.horizontal_displacement.numerics import STOKES_HATS, \
	 RS, DB_ST, DB_R, X_0S
from examples.linear_wave.horizontal_displacement.numerics import OUT_FILE \
	 as IN_FILE

KEYS = ['delta_x', 'x_0', 'Sthat', 'St', 'R', 'mean', 'max', 'min']
OUT_FILE = '../../data/linear_wave/displacement_analysis.csv'

def main():
	"""
	Compute the horizontal displacement with and without history effects.

	The horizontal displacement between a simulation performed with history
	effects and without history effects is computed for particles of varying
	sizes (Stokes numbers) and densities.
	"""
	numerics = pd.read_csv(IN_FILE)
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	for stokes_hat in STOKES_HATS:
		delta_xfs = [] # list to store results for multiple x0s for each Sthat
		st = stokes_hat * (1 / DB_R - 0.5)
		for x_0 in X_0S:
			# extract data
			params = {'x_0': x_0, 'Sthat': stokes_hat, 'R': DB_R,
					  'history': True}
			x_h, z_h = extract_data(['x', 'z'], numerics, params)
			x_h, z_h = x_h.to_numpy(), z_h.to_numpy()
			xh_cross, zh_cross = find_midpoints(x_h, z_h)
			coeffs, _ = curve_fit(f, zh_cross, xh_cross)
			a, b, c, d = coeffs
			zh_curve = np.linspace(z_h[0], z_h[-1], len(z_h))
			xh_curve = f(zh_curve, a, b, c, d)

			params['history'] = False
			x, z = extract_data(['x', 'z'], numerics, params)
			x, z = x.to_numpy(), z.to_numpy()
			x_cross, z_cross = find_midpoints(x, z)
			coeffs, _ = curve_fit(f, z_cross, x_cross)
			a, b, c, d = coeffs
			z_curve = np.linspace(z[0], z[-1], len(z))
			x_curve = f(z_curve, a, b, c, d)

			# compute horizontal displacement
			delta_x_f = compute_horizontal_displacement(x_0, x_curve[-1],
														xh_curve[-1])
			delta_xfs.append(delta_x_f)

		# compute statistics and store results
		results = update_results(results, [np.array(delta_xfs), np.array(X_0S)],
								[stokes_hat, st, DB_R, np.mean(delta_xfs),
								 np.max(delta_xfs), np.min(delta_xfs)])
	for r in RS:
		delta_xfs = [] # list to store results for multiple x0s for each R
		st = DB_ST * (1 / r - 0.5)
		for x_0 in X_0S:
			# extract data
			params = {'x_0': x_0,'Sthat': DB_ST, 'R': r, 'history': True}
			x_h, z_h = extract_data(['x', 'z'], numerics, params)
			x_h, z_h = x_h.to_numpy(), z_h.to_numpy()
			xh_cross, zh_cross = find_midpoints(x_h, z_h)
			coeffs, _ = curve_fit(f, zh_cross, xh_cross)
			a, b, c, d = coeffs
			zh_curve = np.linspace(z_h[0], z_h[-1], len(z_h))
			xh_curve = f(zh_curve, a, b, c, d)

			params['history'] = False
			x, z = extract_data(['x', 'z'], numerics, params)
			x, z = x.to_numpy(), z.to_numpy()
			x_cross, z_cross = find_midpoints(x, z)
			coeffs, _ = curve_fit(f, z_cross, x_cross)
			a, b, c, d = coeffs
			z_curve = np.linspace(z[0], z[-1], len(z))
			x_curve = f(z_curve, a, b, c, d)

			if x_0 == 0 and r == 0.64067:
				plt.plot(x, z, '-k')
				plt.plot(x_h, z_h, '--k')
				plt.scatter(x_cross, z_cross, ec='k', fc='none', marker='.')
				plt.scatter(xh_cross, zh_cross, ec='k', fc='none', marker='.')
				plt.plot(x_curve, z_curve, '-r')
				plt.plot(xh_curve, zh_curve, '-r')

			# compute horizontal displacement
			delta_x_f = compute_horizontal_displacement(x_0, x_curve[-1],
														xh_curve[-1])
			delta_xfs.append(delta_x_f)

		# compute statistics and store results
		results = update_results(results, [np.array(delta_xfs), np.array(X_0S)],
								[DB_ST, st, r, np.mean(delta_xfs),
								 np.max(delta_xfs), np.min(delta_xfs)])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)
	plt.show()

def find_midpoints(x, z):
	"""Find the horizontal midpoints of orbits in a particle trajectory."""
	right, left = [], []
	for i in range(1, len(x) - 1):
		if x[i - 1] < x[i] and x[i + 1] < x[i]: right.append(i)
		if x[i] < x[i - 1] and x[i] < x[i + 1]: left.append(i)
	if len(left) < len(right): right = right[:-1]
	mid = np.rint((np.array(right) + np.array(left)) / 2).astype(int)
	return x[mid], z[mid]

def f(z, a, b, c, d): return a + b * z + c * z * z + d * z ** 3

def compute_horizontal_displacement(x_0, x_f, x_f_history):
	"""
	Return the difference in displacement between two particle positions.

	Parameters
	----------
	x_f : float
		The final horizontal position of a particle without history effects.
	x_f_history : float
		The final horizontal position of a particle with history effects.

	Returns
	-------
	float
		The horizontal displacement.
	"""
	return np.round(np.abs((x_f_history - x_0) - (x_f - x_0))
				  / np.abs(x_f - x_0), 5)

if __name__ == '__main__': main()
