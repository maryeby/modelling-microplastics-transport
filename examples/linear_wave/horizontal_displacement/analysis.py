import pandas as pd
import numpy as np
from scipy.stats import mode

from utils.data_tools import extract_data, update_results
from utils.colors import print_failure
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 DEPTH, WAVELENGTH
from examples.linear_wave.horizontal_displacement.numerics import STOKES_HATS, \
	 RS, DB_ST, DB_R, X_0S
from examples.linear_wave.horizontal_displacement.numerics import OUT_FILE \
	 as IN_FILE

KEYS = ['percent_displacement', 'x_0', 'Sthat', 'R', 'mean', 'median',
		'stdev', 'max', 'min']
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

	for stokes_hat in STOKES_HATS:
		delta_xfs = [] # list to store results for multiple x0s for each St
		for x_0 in X_0S:
			# extract data
			params = {'x_0': x_0, 'Sthat': stokes_hat, 'R': DB_R,
					  'history': False}
			x = extract_data('x', numerics, params).to_numpy()
			params['history'] = True
			x_h = extract_data('x', numerics, params).to_numpy()

			# compute horizontal displacement
			delta_x_f = compute_horizontal_displacement(x_0, x[-1], x_h[-1])
			delta_xfs.append(delta_x_f)

		# compute statistics and store results
		results = update_results(results, [np.array(delta_xfs), np.array(X_0S)],
								[stokes_hat, DB_R, np.mean(delta_xfs),
								np.median(delta_xfs), np.std(delta_xfs),
								np.max(delta_xfs), np.min(delta_xfs)])
	for r in RS:
		delta_xfs = [] # list to store results for multiple x0s for each R
		for x_0 in X_0S:
			# extract data
			params = {'x_0': x_0,'St': DB_ST, 'R': r, 'history': False}
			x = extract_data('x', numerics, params).to_numpy()
			params['history'] = True
			x_h = extract_data('x', numerics, params).to_numpy()

			# compute horizontal displacement
			delta_x_f = compute_horizontal_displacement(x_0, x[-1], x_h[-1])
			delta_xfs.append(delta_x_f)

		# compute statistics and store results
		results = update_results(results, [np.array(delta_xfs), np.array(X_0S)],
								[DB_ST, r, np.mean(delta_xfs),
								 np.median(delta_xfs), np.std(delta_xfs),
								 np.max(delta_xfs), np.min(delta_xfs)])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def compute_horizontal_displacement(x_0, x_f, x_f_history):
	"""
	Return the horizontal displacement between two final positions.

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
	return np.abs((x_f_history - x_0) - (x_f - x_0)) / np.abs(x_f - x_0)

if __name__ == '__main__': main()
