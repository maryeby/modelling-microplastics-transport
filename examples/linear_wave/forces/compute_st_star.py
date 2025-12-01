import numpy as np
import pandas as pd
from intersect import intersection
from itertools import product

from utils.data_tools import extract_data, update_results
from examples.linear_wave.forces.numerics import STOKES_HATS, RS
from examples.linear_wave.forces.analysis import OUT_FILE as IN_FILE

TOL = 1e-1
KEYS = ['Sthat*', 'Sthat75', 'St*', 'St75', 'R']
OUT_FILE = '../../data/linear_wave/forces_st_star.csv'

def main():
	r"""
	Compute $St^*$, where the history force is equal to the Stokes drag.

	The influence of the history force on a particle in a linear wave is equal
	to that of the Stokes drag when the amplitude of the horizontal
	component of the history force intersects the amplitude of the horizontal
	component of the Stokes drag, as shown in subplot *(a)* of the figure
	produced by `plot_coefficients.py`. This intersection value is referred to
	as $St^*$, and is computed for various particle densities. The point at
	which the history force is 75% of the Stokes drag is also computed,
	$St_{75}$.
	"""
	analysis = pd.read_csv(IN_FILE)
	results = {key: [] for key in KEYS}
	for r in RS:
		# extract data
		params = {'R': r, 'force': 'stokes_drag'}
		stokes_drag = extract_data('A', analysis, params)
		params['force'] = 'history_force'
		history, st = extract_data(['A', 'St'], analysis, params)
		st, stokes_drag, history = st.to_numpy(), stokes_drag.to_numpy(), \
								   history.to_numpy()
		# compute St*
		st_star, a_star, i, _ = intersection(st, stokes_drag, st, history)
		sthat_star, ahat_star, i, _ = intersection(STOKES_HATS, stokes_drag,
												   STOKES_HATS, history)
		st_star = st_star[0] if 0 < len(st_star) else 0
		a_star = a_star[0] if 0 < len(a_star) else None
		sthat_star = sthat_star[0] if 0 < len(sthat_star) else 0
		ahat_star = ahat_star[0] if 0 < len(ahat_star) else None
		i = int(np.rint(i[0])) if 0 < len(i) else None

		# compute St75
		st75 = np.interp(0.75, history[:i] / stokes_drag[:i], st[:i]) if i \
			   and 0 < i and not np.isclose(np.interp(0.75,
			   history[:i] / stokes_drag[:i], st[:i]), st[0]) else 0
		sthat75 = np.interp(0.75, history[:i] / stokes_drag[:i],
				  STOKES_HATS[:i]) if i and 0 < i \
				  and not np.isclose(np.interp(0.75,
				  history[:i] / stokes_drag[:i], STOKES_HATS[:i]),
				  STOKES_HATS[0]) else 0

		# store results
		if st_star == 0 and st75 == 0:
			results = update_results(results, [], [0, 0, 0, 0, 0])
		else:
			results = update_results(results, [], [sthat_star, sthat75, st_star,
												   st75, r])
	# write results to data file
	df = pd.DataFrame(results)
	df.replace(0, pd.NA, inplace=True)
	df.dropna(how='all', inplace=True)
	df.to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()
