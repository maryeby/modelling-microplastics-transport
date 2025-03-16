import warnings
import numpy as np
import pandas as pd
from itertools import repeat
from parallelbar import progress_starmap

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# particle conditions
STOKES_NUMS = np.round(np.linspace(0.01, 0.05, 8, endpoint=False).tolist() \
			+ np.arange(0.05, 1.05, 0.05).tolist(), 3)
X_0, Z_0 = 0, 0

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1

# simulation conditions
SCALE = 2 / 3
BETA = 0.99
R = SCALE * BETA
NUM_PERIODS = 3
DELTA_T = 5e-3
NUM_TASKS = len(STOKES_NUMS)
INCLUDE_HISTORY = True
HIDE_PROGRESS = True
KEYS = ['t', 'x', 'z', 'xdot', 'zdot', 'fluid_pressure_gradient_x',
		'fluid_pressure_gradient_z', 'buoyancy_force_x', 'buoyancy_force_z',
		'added_mass_force_x', 'added_mass_force_z', 'stokes_drag_x',
		'stokes_drag_z', 'history_force_x', 'history_force_z', 'St']
OUT_FILE = '../../data/water_wave/forces_numerics.csv'

def main():
	"""
	Simulate negatively buoyant particles transported through a wave.

	Simulations are performed for particles of varying sizes (Stokes numbers),
	and history effects are included. Results are saved to the `data/water_wave`
	directory.
	"""
	# create dictionary to store results, initialize variables for simulations
	warnings.filterwarnings('ignore')
	results = {key: [] for key in KEYS}
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, NUM_PERIODS, DELTA_T)

	# run simulations in parallel
	params = zip(STOKES_NUMS, repeat(wave), repeat(t), repeat(y))
	sols = progress_starmap(run_simulation, params, n_cpu=4, total=NUM_TASKS)
	for sol in sols: results = update_results(results, sol[:-1], [sol[-1]])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def run_simulation(stokes_num, wave, t, y):
	"""
	Run a numerical simulation with the specified conditions.

	Parameters
	----------
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	wave : Wave (obj)
		The wave through which the particle is transported.
	t : ndarray
		1D array of `float` time series data.
	y : list
		A list of `float` data, the initial particle position and velocity.

	Returns
	-------
	list
		List containing `ndarray` elements of `float` data.
	"""
	n = -3
	particle = prt.Particle(stokes_num)
	system = ts.MyTransportSystem(particle, wave, R)
	x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, mass_z, \
	   drag_x, drag_z, history_x, history_z = system.maxey_riley(t, y,
											  INCLUDE_HISTORY, HIDE_PROGRESS)
	sols = [t, x, z, xdot, zdot, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x,
			mass_z, drag_x, drag_z, history_x, history_z]
	sols = [sol[:n] for sol in sols]
	sols.append(stokes_num)
	return sols

if __name__ == '__main__':
	main()
