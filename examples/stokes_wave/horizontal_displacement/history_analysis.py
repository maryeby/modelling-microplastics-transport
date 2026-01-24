import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.optimize import curve_fit

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, update_results, NEUTRAL_R
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 DEPTH, WAVELENGTH
from examples.linear_wave.horizontal_displacement.numerics import STOKES_HATS, \
	 RS, DB_ST, DB_R, X_0S
from examples.stokes_wave.horizontal_displacement.numerics import OUT_FILE \
	 as IN_FILE

NEGATIVE_ZLIM, POSITIVE_ZLIM = -2, -0.45
KEYS = ['delta_x', 'x_0', 'Sthat', 'St', 'R', 'mean', 'max', 'min']
OUT_FILE = '../../data/stokes_wave/history_displacement_analysis.csv'

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
		sol = compute_displacement(numerics, sthat, DB_R, results)
		if sol: results = update_results(results, sol[:2], sol[2:])
	for r in RS:
		sol = compute_displacement(numerics, DB_ST, r, results)
		if sol: results = update_results(results, sol[:2], sol[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)
	plt.show()

def compute_displacement(numerics, sthat, r, results):
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
	results : dict
		The results dictionary to update.

	Returns
	-------
	results : dict
		The updated dictionary of results.
	"""
	st = sthat * (1 / r - 0.5)
	delta_xfs = []
	for x_0 in X_0S:
		x_finals = []
		for h in [False, True]:
			# extract data
			params = {'x_0': x_0, 'Sthat': sthat, 'R': r, 'history': h}
			x, z = extract_data(['x', 'z'], numerics, params)
			x, z = x.to_numpy(), z.to_numpy()

			# fit curves to the data and find the mean final position
			position = 'bottom' if r < NEUTRAL_R else 'top'
			x_cross, z_cross = find_midpoints(x, z, position)
			if 4 <= len(x_cross):
				x_curve, z_curve = fit_curve(x_cross, z_cross, z[0], z[-1])
				if r < NEUTRAL_R and z_curve[-1] < NEGATIVE_ZLIM:
					n = np.argmax(z_curve < NEGATIVE_ZLIM)
					x_finals.append(x_curve[n - 1])
				elif NEUTRAL_R < r and POSITIVE_ZLIM < z_curve[-1]:
					n = np.argmax(POSITIVE_ZLIM < z_curve)
					x_finals.append(x_curve[n - 1])
				else:
					n = -1
					x_finals.append(x_curve[-1])

				# plot a specific case
				if x_0 == 0 and sthat == DB_ST and r == 0.68533:
					fmt = '--k' if h else '-k'
					plt.scatter(x_cross, z_cross, ec='k', fc='none', marker='.')
					plt.plot(x, z, fmt)
					plt.plot(x_curve[:n], z_curve[:n], '-m')

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

def find_midpoints(x, z, position):
	"""
	Find the horizontal midpoints of orbits in a particle trajectory.
	
	Parameters
	----------
	x, z : ndarray
		The horizontal and vertical position(s) of the particle.
	position : str
		Where in the orbits to take the midpoints: `top`, `mid`, or `bottom`.

	Returns
	-------
	ndarray
		1D arrays of horizontal and vertical midpoint positions.
	"""
	right, left, top, bottom = [], [], [], []
	for i in range(1, len(x) - 1):
		if x[i - 1] < x[i] and x[i + 1] < x[i]: right.append(i)
		if x[i] < x[i - 1] and x[i] < x[i + 1]: left.append(i)
		if z[i - 1] < z[i] and z[i + 1] < z[i]: top.append(i)
		if z[i] < z[i - 1] and z[i] < z[i + 1]: bottom.append(i)
	if len(left) < len(right): right = right[:-1]
	mid = np.rint((np.array(right) + np.array(left)) / 2).astype(int)
	if position == 'top':
		return x[top], z[top]
	elif position == 'mid':
		return x[mid], z[mid]
	else:
		return x[bottom], z[bottom]

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
