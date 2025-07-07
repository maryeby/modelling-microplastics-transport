import pandas as pd
import numpy as np
from scipy.stats import mode

from utils.data_tools import extract_data, update_results
from utils.colors import print_failure
from examples.water_wave.horizontal_displacement.dibenedetto_numerics import \
	 DEPTH, WAVELENGTH
from examples.water_wave.horizontal_displacement.numerics import STOKES_NUMS, \
	 BETAS, DB_ST, DB_BETA, X_0S
from examples.water_wave.horizontal_displacement.numerics import OUT_FILE \
	 as IN_FILE

OUT_FILE = '../../data/water_wave/displacement_analysis.csv'

def main():
	"""
	Compute the horizontal displacement with and without history effects.

	The percent horizontal displacement between a simulation performed with
	history effects and without history effects is computed for particles of
	varying sizes (Stokes numbers) and densities.
	"""
	numerics = pd.read_csv(IN_FILE)
	keys = ['percent_displacement', 'x_0', 'St', 'beta', 'mean', 'median',
			'stdev', 'max', 'min']
	results = {key: [] for key in keys}

	for stokes_num in STOKES_NUMS:
		delta_xfs = [] # list to store results for multiple x0s for each St
		for x_0 in X_0S:
			# extract data
			params = {'x_0': x_0, 'St': stokes_num, 'beta': DB_BETA,
					  'history': False}
			x = extract_data('x', numerics, params).to_numpy()
			params['history'] = True
			x_h = extract_data('x', numerics, params).to_numpy()

			# compute horizontal displacement
			delta_x_f = compute_horizontal_displacement(x_0, x[-1], x_h[-1])
			delta_xfs.append(delta_x_f)

		# compute statistics and store results
		results = update_results(results, [np.array(delta_xfs), np.array(X_0S)],
								[stokes_num, DB_BETA, np.mean(delta_xfs),
								np.median(delta_xfs), np.std(delta_xfs),
								np.max(delta_xfs), np.min(delta_xfs)])
	for beta in BETAS:
		delta_xfs = [] # list to store results for multiple x0s for each beta
		for x_0 in X_0S:
			# extract data
			params = {'x_0': x_0,'St': DB_ST, 'beta': beta, 'history': False}
			x = extract_data('x', numerics, params).to_numpy()
			params['history'] = True
			x_h = extract_data('x', numerics, params).to_numpy()

			# compute horizontal displacement
			delta_x_f = compute_horizontal_displacement(x_0, x[-1], x_h[-1])
			delta_xfs.append(delta_x_f)

		# compute statistics and store results
		results = update_results(results, [np.array(delta_xfs), np.array(X_0S)],
								[DB_ST, beta, np.mean(delta_xfs),
								 np.median(delta_xfs), np.std(delta_xfs),
								 np.max(delta_xfs), np.min(delta_xfs)])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def compute_horizontal_displacement(x_0, x_f, x_f_history):
	"""
	Return the percent horizontal displacement between two final positions.

	Parameters
	----------
	x_f : float
		The final horizontal position of a particle without history effects.
	x_f_history : float
		The final horizontal position of a particle with history effects.

	Returns
	-------
	float
		The percent horizontal displacement.
	"""
	return (x_f_history - x_0) * 100 / (x_f - x_0) - 100

if __name__ == '__main__': main()
