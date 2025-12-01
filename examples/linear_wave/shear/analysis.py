import warnings
import numpy as np
import pandas as pd
from itertools import repeat
from parallelbar import progress_starmap

from utils.data_tools import extract_data, update_results
from examples.linear_wave.shear.numerics import X_0S, RADIUS, ANGLE, \
	 Z_NEGATIVE, Z_POSITIVE, ST_TO_SHOW, R_TO_SHOW, STOKES_HATS, RS, \
	 NUM_POINTS, NUM_TASKS
from examples.linear_wave.shear.numerics import OUT_FILE as IN_FILE

NUM_CPUS = None
TIMEOUT = 240
KEYS = ['x', 'z', 'shear', 'x_0', 'z_0', 'St', 'beta', 'history']
OUT_FILE1 = IN_FILE # update input file
OUT_FILE2 = '../../data/linear_wave/orbit_shear_difference.csv'

def main():
	"""
	Compute the shearing of the orbits in particle trajectories.

	The orbit shearing is computed for particles of varying sizes and densities
	transported through a linear wave of arbitrarily deep water. The difference
	in total shear between simulations with and without history is also
	computed, and results are saved to the `data/linear_wave` directory.
	"""
	# create dict to store sols and read numerics data
	results = {key: [] for key in KEYS}
	df = pd.read_csv(IN_FILE)

	# parameters for parallel processing
	repeated_st = (STOKES_HATS.tolist() + [ST_TO_SHOW] * len(RS)) * 2
	repeated_r = ([R_TO_SHOW] * len(STOKES_HATS) + RS.tolist()) * 2
	repeated_history = [False] * len(repeated_st) + [True] * len(repeated_st)
	point_index = []
	for i in range(NUM_POINTS):
		point_index += [i] * (len(STOKES_HATS) + len(RS)) * 2
	repeated_st *= NUM_POINTS
	repeated_beta *= NUM_POINTS
	repeated_history *= NUM_POINTS
	params = zip(repeat(df), repeated_st, repeated_r, point_index,
				 repeated_history)

	# compute orbit shearing in parallel and store solutions
	sols = progress_starmap(compute_shear, params, n_cpu=NUM_CPUS,
							total=NUM_TASKS, process_timeout=TIMEOUT)
	for sol in sols: results = update_results(results, sol[:3], sol[3:])
	df = pd.DataFrame(results)
	df.to_csv(OUT_FILE1, index=False) # update data file

	# compute the difference in shear between sols with and without history
	repeated_st = STOKES_HATS.tolist() + [ST_TO_SHOW] * len(RS)
	repeated_r = [R_TO_SHOW] * len(STOKES_HATS) + RS.tolist()
	results = {'shear_difference': [], 'Sthat': [], 'R': []}
	params = zip(repeat(df), repeated_st, repeated_r)
	sols = progress_starmap(shear_difference, params, n_cpu=NUM_CPUS,
							total=NUM_TASKS // (2 * NUM_POINTS),
							process_timeout=TIMEOUT)
	for sol in sols: update_results(results, [], sol)
	pd.DataFrame(results).to_csv(OUT_FILE2, index=False) # save to data file

def compute_shear(df, stokes_hat, r, i, include_history):
	r"""
	Compute the shear of an orbit over time from the specified initial point.

	Parameters
	----------
	df : DataFrame
		A `DataFrame` contianing `float` elements of particle position data.
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	r : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether to include history effects.

	Returns
	-------
	ndarray
		A 1D array of `float` elements containing the shear for each period and
		total shear.

	Notes
	-----
	The shear is computed as the norm of the composition matrix of the
	horizontal shear $m_x$ and vertical shear $m_z$ of the orbit throughout the
	particle trajectory,
	$$\begin{Vmatrix}
	1 + m_x m_z & m_x\\
	m_z & 1
	\end{Vmatrix}.$$
	"""
	shear = []
	z_center = Z_NEGATIVE * RADIUS if r < 1 else Z_POSITIVE * RADIUS
	z_0s = np.round(RADIUS * np.sin(ANGLE) + z_center, 3)
	xn, zn = extract_data(['x', 'z'], df, {'history': include_history,
			'x_0': X_0S[i], 'z_0': z_0s[i], 'Sthat': stokes_hat, 'r': r})
	x, z = np.abs(xn).tolist(), np.abs(zn).tolist()
	
	# compute shear between orbits
	for j in range(len(x) - 1):
		mx = (x[j + 1] - x[j]) / z[j + 1] # x shear factor
		mz = (z[j + 1] - z[j]) / x[j + 1] # z shear factor
		shear.append(np.linalg.norm([[1 + mx * mz, mx],[mz, 1]]))

	# compute total shear
	mx = (x[-1] - x[0]) / z[-1] # x shear factor
	mz = (z[-1] - z[0]) / x[-1] # z shear factor
	shear.append(np.linalg.norm([[1 + mx * mz, mx],[mz, 1]]))

	return xn, zn, np.array(shear), X_0S[i], z_0s[i], stokes_num, beta, \
		   include_history

def shear_difference(df, stokes_num, beta):
	"""Compute the shear difference between solutions with & without history."""
	params = {'Sthat': stokes_hat, 'R': r, 'history': False}
	shear_list, shear_h_list = [], []
	z_center = Z_NEGATIVE * RADIUS if r < 2 / 3 else Z_POSITIVE * RADIUS
	z_0s = np.round(RADIUS * np.sin(ANGLE) + z_center, 3)

	for i in range(NUM_POINTS):
		params['x_0'], params['z_0'] = X_0S[i], z_0s[i]
		shear = extract_data('shear', df, params).to_numpy()[-1]
		shear_list.append(shear)
		params['history'] = True
		shear = extract_data('shear', df, params).to_numpy()[-1]
		shear_h_list.append(shear)

	shear_diff = np.round(np.mean(np.array(shear_list)
								- np.array(shear_h_list)), 5)
	x = extract_data('x', df, params)
	return [shear_diff, stokes_num, beta]

if __name__ == '__main__':
	main()
