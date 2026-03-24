import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.optimize import curve_fit

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, update_results
from utils.colors import print_failure
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

	for sthat in STOKES_HATS:
		sol = compute_displacement(numerics, sthat, DB_R)
		if sol: results = update_results(results, sol[:2], sol[2:])
	for r in RS:
		sol = compute_displacement(numerics, DB_ST, r)
		if sol: results = update_results(results, sol[:2], sol[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)
	plt.show()

def compute_displacement(numerics, sthat, r):
	r"""
	Return the difference in total horizontal displacement between 2 particles.

	Parameters
	----------
	numerics : DataFrame
		A DataFrame containing the numerical solutions.
	sthat : float
		The density-independent Stokes number $\widehat{St}$.
	r : float
		The ratio between the particle and fluid densities.

	Returns
	-------
	list
		A list of elements to add to the results dictionary.
	"""
	st = sthat * (1 / r - 0.5)
	delta_xfs = []
	for x_0 in X_0S:
		x_finals = []
		for history in [False, True]:
			# extract data
			params = {'x_0': x_0, 'Sthat': sthat, 'R': r, 'history': history}
			x, z = extract_data(['x', 'z'], numerics, params)
			x, z = x.to_numpy(), z.to_numpy()

			# fit curves to the data and find the mean final position
			x_cross, z_cross = find_midpoints(x, z)
			if 4 <= len(x_cross):
				x_curve, z_curve = fit_curve(x_cross, z_cross, z[0], z[-1])
				x_finals.append(x_curve[-1])

				# plot a specific case
				if x_0 == 0 and sthat == 0.9054 and r == DB_R:
					fmt = '--k' if history else '-k'
					plt.scatter(x_cross, z_cross, ec='k', fc='none', marker='.')
					plt.plot(x, z, fmt)
					plt.plot(x_curve, z_curve, '-m')

		# compute the total difference in horizontal displacement
		if len(x_finals) == 2:
			delta_xfs.append(np.round(np.abs((x_finals[0] - x_0) - (x_finals[1]
							 - x_0)) / np.abs(x_finals[0] - x_0), 5))
	# update results
	if len(delta_xfs) == len(X_0S):
		return [np.array(delta_xfs), np.array(X_0S), sthat, st, r,
				np.mean(delta_xfs), np.max(delta_xfs), np.min(delta_xfs)]
	else:
		return []

def find_midpoints(x, z):
	"""Find the horizontal midpoints of orbits in a particle trajectory."""
	right, left = [], []
	for i in range(1, len(x) - 1):
		if x[i - 1] < x[i] and x[i + 1] < x[i]: right.append(i)
		if x[i] < x[i - 1] and x[i] < x[i + 1]: left.append(i)
	if len(left) < len(right): right = right[:-1]
	mid = np.rint((np.array(right) + np.array(left)) / 2).astype(int)
	return x[mid], z[mid]

def fit_curve(x_cross, z_cross, z_0, z_f):
	r"""
	Return a curve fit to the provided data.

	Parameters
	----------
	x_cross, z_cross : ndarray
		1D arrays of `float` data, the horizontal and vertical positions.
	z_f : float
		The final vertical position of the particle.

	Returns
	-------
	x_curve, z_curve : ndarray
		1D arrays of `float` data, comprising the curve.

	Notes
	-----
	The function used to fit the curve to the data is,
	$$f(z) = a + bz + cz^2 + dz^3.$$
	"""
	coeffs, _ = curve_fit(f, z_cross, x_cross)
	a, b, c, d = coeffs
	z_curve = np.linspace(z_0, z_f, 1000)
	x_curve = f(z_curve, a, b, c, d)
	return x_curve, z_curve

def f(z, a, b, c, d): return a + b * z + c * z * z + d * z ** 3

if __name__ == '__main__': main()
