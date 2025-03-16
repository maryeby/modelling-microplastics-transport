import warnings
import pandas as pd
import numpy as np

from utils.data_tools import update_results
from transport_framework import particle as prt 
from models import deep_water_wave as fl
from models import haller_system as hts
from models import my_system as ts
from examples.deep_water_wave.inertial_numerics import AMPLITUDE, WAVELENGTH, \
	 STOKES_NUM, X_0, Z_0, R, NUM_PERIODS, INCLUDE_HISTORY

OUT_FILE = '../data/deep_water_wave/global_error.csv'

def main():
	"""
	Compute the global error of our model compared to numerical integration.

	The global error is computed between numerically integrated solutions and
	solutions produced with the multi-step integration method.[^1] The
	simulations model a negatively buoyant particle in a linear wave of
	infinitely deep water without history effects. Results are saved to the
	`data/deep_water_wave` directory.

	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order rical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# create dictionary to store solutions and array of timesteps
	results = {'global_error': [], 'delta_t': [], 'order': []}
	timesteps = np.linspace(1e-3, 1e-1, 10, endpoint=False)

	# create Wave, Particle, and TransportSystem objects
	wave = fl.DeepWaterWave(AMPLITUDE, WAVELENGTH)
	particle = prt.Particle(STOKES_NUM * wave.froude_num * R)
	system = ts.MyTransportSystem(particle, wave, R)
	h_system = hts.HallerTransportSystem(particle, wave, R)

	# set initial particle position and velocity
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	warnings.filterwarnings('ignore')

	i, total = 1, len(timesteps) * 3
	for delta_t in timesteps:
		# run numerical integration (using the HallerTransportSystem)
		print(f'Numerically integrating (delta_t = {delta_t:.2e})...', end='')
		x, z, _, _, _ = h_system.run_numerics(h_system.maxey_riley, X_0,
			 								  Z_0, NUM_PERIODS, delta_t)
		haller = np.array([x, z]).T
		print('done.')

		# run simulation using multi-step integration scheme (MyTransportSystem)
		t = np.arange(0, NUM_PERIODS * wave.period + delta_t, delta_t)
		for order in [1, 2, 3]:
			print(f'\n({i}/{total:g}) Running simulation with delta_t =',
				  f'{delta_t:.2e} and order {order}...')
			x, z, _, _, _, _, _, _, _, _, _, _, _, _, \
			   _ = system.maxey_riley(t, y, include_history=False,
										 order=order)
			sol = np.array([x, z]).T
			i += 1

			# compute global error and store solutions
			global_error = np.linalg.norm(haller - sol, axis=1).max()
			results = update_results(results, [],
									[global_error, delta_t, order])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()
