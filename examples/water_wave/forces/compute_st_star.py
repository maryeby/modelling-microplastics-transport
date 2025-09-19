import numpy as np
import pandas as pd
from intersect import intersection
from itertools import product

from utils.data_tools import extract_data, update_results
from examples.water_wave.forces.numerics import STOKES_NUMS, BETAS, RS, EPSILONS
from examples.water_wave.forces.analysis import OUT_FILE as IN_FILE

TOL = 1e-1
KEYS = ['St*', 'St75', 'epsilon', 'beta', 'R']
OUT_FILE = '../../data/water_wave/forces_st_star.csv'

def main():
	r"""
	Compute $St^*$, where the history force is equal to the Stokes drag.

	The influence of the history force on a particle in a linear wave is equal
	to that of the Stokes drag when the amplitude of the horizontal
	component of the history force intersects the amplitude of the horizontal
	component of the Stokes drag, as shown in subplot *(a)* of the figure
	produced by `plot_coefficients.py`. This intersection value is referred to
	as $St^*$, and is computed for various values of the wave steepness
	$\epsilon$ and various particle densities. The point at which the history
	force is 75% of the Stokes drag is also computed, $St_{75}$.
	"""
	analysis = pd.read_csv(IN_FILE)
	results = {key: [] for key in KEYS}
	for r, epsilon in product(RS, EPSILONS):
		# extract data
		params = {'R': r, 'epsilon': epsilon, 'force': 'stokes_drag'}
		stokes_drag = extract_data('A', analysis, params)
		params['force'] = 'history_force'
		history, st = extract_data(['A', 'St'], analysis, params)
		st, stokes_drag, history = st.to_numpy(), stokes_drag.to_numpy(), \
								   history.to_numpy()
		# compute St*
		st_star, a_star, i, _ = intersection(st, stokes_drag, st, history)
		st_star = st_star[0] if 0 < len(st_star) else 0
		a_star = a_star[0] if 0 < len(a_star) else None
		i = int(np.rint(i[0])) if 0 < len(i) else None

		# compute St75
		st75 = np.interp(0.75, history[:i] / stokes_drag[:i], st[:i]) if i \
			   and 0 < i and not np.isclose(np.interp(0.75,
			   history[:i] / stokes_drag[:i], st[:i]), STOKES_NUMS[0]) else 0

		# store results
		beta = BETAS[np.where(RS == r),][0][0]
		if st_star == 0 and st75 == 0:
			results = update_results(results, [], [0, 0, -1, 0, 0])
		else:
			results = update_results(results, [], [st_star, st75, epsilon, beta,
																			 r])
	# compute St hat and write to data file
	results['Sthat*'] = (np.array(results['St*']) \
					  / np.array(results['epsilon'])).tolist()
	results['Sthat75'] = (np.array(results['St75']) \
					   / np.array(results['epsilon'])).tolist()
	df = pd.DataFrame(results)
	df.replace(0, pd.NA, inplace=True)
	df.replace(-1, pd.NA, inplace=True)
	df.dropna(how='all', inplace=True)
	df.to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()
