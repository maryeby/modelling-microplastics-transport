import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import product

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, update_results, NEUTRAL_R
from examples.linear_wave.horizontal_displacement.numerics import DB_ST, DB_R
from examples.bichromatic_wave.trajectories.numerics import SLOPE
from examples.bichromatic_wave.horizontal_displacement.numerics import X_0S, \
	 STOKES_HATS, RS
from examples.bichromatic_wave.horizontal_displacement.numerics import OUT_FILE\
	 as IN_FILE

KEYS = ['delta_x', 'x_0', 'Sthat', 'St', 'R', 'slope', 'mean', 'max', 'min']
OUT_FILE = '../../data/bichromatic_wave/displacement_analysis.csv'
#TODO: change the computation of displacement for sloped bed to use pythagorean theorem

def main():
	"""
	Compute the displacement between simulations with and without history.

	The horizontal displacement between a simulation performed with history vs
	without history is computed for particles of varying sizes
	(Stokes numbers) and densities.
	"""
	numerics = pd.read_csv(IN_FILE)
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	for sthat, slope in product(STOKES_HATS, [0, SLOPE]):
		sol = compute_displacement(numerics, sthat, DB_R, slope, results)
		if sol: results = update_results(results, sol[:2], sol[2:])
	for r, slope in product(RS, [0, SLOPE]):
		sol = compute_displacement(numerics, DB_ST, r, slope, results)
		if sol: results = update_results(results, sol[:2], sol[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def compute_displacement(numerics, sthat, r, slope, results):
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
	slope : float
		The slope of the seabed.
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
			params = {'x_0': x_0, 'Sthat': sthat, 'R': r, 'history': h,
					  'slope': slope}
			x, z = extract_data(['x', 'z'], numerics, params)
			x, z = x.to_numpy(), z.to_numpy()
			if np.any(x):
				x_finals.append(x[-1])

		if len(x_finals) == 2:
			# compute the total difference in horizontal displacement
			delta_xfs.append(np.round(np.abs((x_finals[0] - x_0) - (x_finals[1]
							 - x_0)) / np.abs(x_finals[0] - x_0), 5))
	if 0 < len(delta_xfs):
		# update results
		return [np.array(delta_xfs), np.array(X_0S), sthat, st, r, slope,
				np.mean(delta_xfs), np.max(delta_xfs), np.min(delta_xfs)]
	else:
		return []

if __name__ == '__main__': main()
