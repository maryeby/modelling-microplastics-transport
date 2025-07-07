import warnings
import pandas as pd
import numpy as np
from itertools import repeat
from parallelbar import progress_starmap

from utils.data_tools import extract_data, update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts
from models.my_system import compute_drift_velocity as find_crossings
from examples.water_wave.critical_nums.analysis import DEPTH, AMPLITUDE, \
	 WAVELENGTH, X_0, SCALE, NUM_PERIODS, DELTA_T

Z_0 = 0
BETAS = np.round([0.83, 0.84, 0.85, 0.8, 0.8, 0.8], 3)
INCLUDE_HISTORY = False
NUM_CPUS = None
NUM_TASKS = len(BETAS)
TIMEOUT = 240
HIDE_PROGRESS = True
IN_FILE = '../../data/water_wave/critical_nums.csv'
OUT_FILE = '../../data/water_wave/critical_trajectories.csv'

def main():
	"""
	Record numerical solutions for trajectories of particles in a linear wave.

	Simulations are run for negatively buoyant particles in linear waves of deep
	water, for the purposes of investigating the non-monotonic behavior of the
	critical density ratio vs critical Stokes number. Results are saved to the
	`data/water_wave` directory.
	"""
	# create dict to store sols and extract Stokes numbers from data file
	analysis = pd.read_csv(IN_FILE)
	keys = ['x', 'z', 'x_crossings', 'z_crossings', 'St', 'beta']
	results = {key: [] for key in keys}
	stokes_nums = []
	for beta in BETAS[:-3]:
		st = extract_data('St_c', analysis,
						 {'beta_c': beta, 'history': INCLUDE_HISTORY})
		if isinstance(st, pd.core.series.Series): stokes_nums.append(st.iloc[0])
	stokes_nums += [0.05, 0.1067, 0.2]

	# create Wave object and compute initial particle velocity
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]

	# initialize remaining variables for the simulations
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	params = zip(repeat(wave), repeat(t), repeat(y), stokes_nums, BETAS)
	warnings.filterwarnings('ignore')

	# run simulations in parallel and store results
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=NUM_TASKS, process_timeout=TIMEOUT)
	for sol in sols: results = update_results(results, sol[:4], sol[4:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(wave, t, y, stokes_num, beta):
	"""
	Run a numerical simulation with the specified *St* and density ratio *R*.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	t : ndarray
		1D array containing `float` time series data.
	y : list
		A list of `float` data, the initial particle position and velocity.
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	beta : float
		The ratio between the particle and fluid densities.

	Returns
	-------
	list
		A list containing the particle positions, *St*, and *R*.
	"""
	density_ratio = beta * SCALE
	particle = prt.Particle(stokes_num)
	system = ts.MyTransportSystem(particle, wave, density_ratio)
	x, z, xdot, _, t = system.maxey_riley(t, y, INCLUDE_HISTORY,
										  HIDE_PROGRESS)[:5]
	x_crossings, z_crossings, _, _, _ = find_crossings(x, z, xdot, t)
	diff = len(x) - len(x_crossings)
	x_crossings = np.array(x_crossings.tolist() + [np.nan] * diff)
	z_crossings = np.array(z_crossings.tolist() + [np.nan] * diff)
	return [x, z, x_crossings, z_crossings, stokes_num, beta]

if __name__ == '__main__': main()
