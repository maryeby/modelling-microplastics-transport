import pandas as pd
import numpy as np
from time import time

from utils.data_tools import extract_data, update_results
from transport_framework import particle as prt 
from models import rotating_flow as fl
from models import rotating_system as ts
from examples.rigid_body_rotation.numerics import R, STOKES_NUM, X_0, Z_0

T_FINAL = 10
IN_FILE = '../data/rigid_body_rotation/analytics.csv'
OUT_FILE = '../data/rigid_body_rotation/global_error.csv'

def main():
	"""
	Compute the global error for a rotating rigid body.

	The global error is computed with varying timestep sizes to reproduce
	results from [1] Figure 4. Results are saved to the
	`data/rigid_body_rotation` directory.
	
	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# read data, initialize delta_ts, and create dictionary to store solutions
	analytics = pd.read_csv(IN_FILE)
	timesteps = analytics['delta_t'].drop_duplicates().iloc[:-1]
	keys = ['global_error', 'delta_t', 'order', 'computation_time']
	results = {key: [] for key in keys}

	# initialize variables for numerical simulations
	particle = prt.Particle(STOKES_NUM)
	flow = fl.RotatingFlow()
	system = ts.RotatingTransportSystem(particle, flow, R)
	xdot_0, zdot_0 = flow.velocity(X_0, Z_0)
	y = [X_0, Z_0, xdot_0, zdot_0]

	i, total = 1, len(timesteps) * 3
	for delta_t in timesteps:
		t = np.arange(0, T_FINAL, delta_t)
		for order in [1, 2, 3]:
			# compute numerics
			print(f'({i}/{total:g}) Computing numerics for delta_t =',
				  f'{delta_t:.0e}...')
			start = time()
			x, z, _, _, _ = system.maxey_riley(t, y, order)
			finish = time()
			computation_time = finish - start
			print(f'Computations for delta_t = {delta_t:.0e} order {order:g}',
				  f'complete.\t\t{finish - start:5.2f}s\n')
			i += 1

			# compute global error and store solutions
			n = len(x)
			numerics = np.array([x, z]).T
			x, z = extract_data(['x', 'z'], analytics, {'delta_t': delta_t})
			exact = np.array([x.iloc[:n], z.iloc[:n]]).T
			global_error = np.linalg.norm(exact - numerics, axis=1).max()
			results = update_results(results, [], [global_error, delta_t, order,
											   computation_time])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()
