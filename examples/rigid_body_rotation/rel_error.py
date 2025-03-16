import pandas as pd
import numpy as np
from utils.data_tools import extract_data, update_results

DELTA_T = 1e-2
IN_FILE1 = '../data/rigid_body_rotation/numerics.csv'
IN_FILE2 = '../data/rigid_body_rotation/analytics.csv'
OUT_FILE = '../data/rigid_body_rotation/rel_error.csv'

def main():
	"""
	Compute the relative error for a rigid rotating body.

	Results reproduce [1] Figure 3, and are saved to the
	`data/rigid_body_rotation` directory.
	
	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# read data and create dictionary to store results
	numerics = pd.read_csv(IN_FILE1)
	analytics = pd.read_csv(IN_FILE2)
	results = {'t': [], 'e_abs': [], 'e_rel': [], 'order': []}

	# get analytical data
	n = len(numerics['x']) // 3
	x, z, t = extract_data(['x', 'z', 't'], analytics, {'delta_t': DELTA_T})
	x = x[:n]
	z = z[:n]
	t = t[:n]
	exact = np.array([x, z]).T

	for order in [1, 2, 3]:
		# get numerical data
		x, z = extract_data(['x', 'z'], numerics, {'order': order})
		numerical = np.array([x, z]).T

		# compute absolute and relative errors, store solutions
		e_abs = np.linalg.norm(exact - numerical, axis=1)
		e_rel = e_abs / np.linalg.norm(exact, axis=1)
		results = update_results(results, [t, e_abs, e_rel], [order])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()
