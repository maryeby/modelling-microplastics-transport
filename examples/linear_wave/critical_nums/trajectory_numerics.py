import warnings
import pandas as pd
import numpy as np
from itertools import repeat
from parallelbar import progress_starmap

from utils.data_tools import extract_data, update_results, print_parameter
from utils.colors import print_failure
from transport_framework import particle as prt
from models import linear_wave as fl
from models import my_system as ts
from models.my_system import compute_drift_velocity as find_crossings
from examples.linear_wave.critical_nums.analysis import DEPTH, AMPLITUDE, \
	 WAVELENGTH, X_0, NUM_PERIODS, DELTA_T
from examples.linear_wave.critical_nums.analysis import OUT_FILE as IN_FILE

Z_0 = 0
R = 0.6
INCLUDE_HISTORY = False
NUM_CPUS = None
TIMEOUT = 120
HIDE_PROGRESS = True
KEYS = ['x', 'z', 'x_crossings', 'z_crossings', 'Sthat', 'St', 'R']
OUT_FILE = '../../data/linear_wave/critical_trajectories.csv'

def main():
	"""
	Record numerical solutions for trajectories of particles in a linear wave.

	Simulations are run for negatively buoyant particles in linear waves of deep
	water, for the purposes of illustrating oscillatory damping of the particle
	trajectories. Results are saved to the `data/linear_wave` directory.
	"""
	# create dict to store sols and extract Stokes numbers from data file
	analysis = pd.read_csv(IN_FILE)
	results = {key: [] for key in KEYS}
	st_hats = [3 / 7, 60 / 7]
	st = extract_data('St_c', analysis, {'R_c': R, 'history': INCLUDE_HISTORY})
	st_hats.insert(1, st.iloc[0])
	for s in st_hats: print_parameter('St', s * (1 / R - 1 / 2))
	if st_hats[2] < st_hats[1]:
		print_failure('Post-critical Stokes number is too small')
		quit()

	# create Wave object and compute initial particle velocity
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]

	# initialize remaining variables for the simulations
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	params = zip(repeat(wave), repeat(t), repeat(y), st_hats)
	warnings.filterwarnings('ignore')

	# run simulations in parallel and store results
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=len(st_hats), process_timeout=TIMEOUT)
	for sol in sols: results = update_results(results, sol[:4], sol[4:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(wave, t, y, stokes_hat):
	r"""
	Run a numerical simulation with the specified $\widehat{St}$.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	t : ndarray
		1D array containing `float` time series data.
	y : list
		A list of `float` data, the initial particle position and velocity.
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.

	Returns
	-------
	list
		A list containing the particle positions, $\widehat{St}$, $St$, and $R$.
	"""
	particle = prt.Particle(stokes_hat)
	system = ts.MyTransportSystem(particle, wave, R)
	x, z, xdot, _, t = system.maxey_riley(t, y, INCLUDE_HISTORY,
										  HIDE_PROGRESS)[:5]
	x_crossings, z_crossings, _, _, _ = find_crossings(x, z, xdot, t)
	diff = len(x) - len(x_crossings)
	x_crossings = np.array(x_crossings.tolist() + [np.nan] * diff)
	z_crossings = np.array(z_crossings.tolist() + [np.nan] * diff)
	return [x, z, x_crossings, z_crossings, stokes_hat, system.stokes_num, R]

if __name__ == '__main__': main()
