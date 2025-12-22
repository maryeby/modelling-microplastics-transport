import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.contrib.itertools import product

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import linear_wave as fl
from models import my_system as ts

# particle conditions
STOKES_HATS = np.round(np.sort(np.linspace(99 / 670, 297 / 1675, 8).tolist() \
			+ np.linspace(0.18, 1, 21).tolist() + [67 / 330]), 5)
X_0, Z_0 = 0, 0

# wave conditions
DEPTH = 10
WAVELENGTH = 1.5
AMPLITUDE = 0.02

# simulation conditions
RS = np.round([0.5, 8 / 15, 17 / 30, 0.6, 0.66], 5)
NUM_PERIODS = 5
DELTA_T = 5e-3
INCLUDE_HISTORY = True
HIDE_PROGRESS = True
KEYS = ['t', 'x', 'z', 'xdot', 'zdot', 'fluid_pressure_gradient_x',
		'fluid_pressure_gradient_z', 'buoyancy_force_x', 'buoyancy_force_z',
		'added_mass_force_x', 'added_mass_force_z', 'stokes_drag_x',
		'stokes_drag_z', 'history_force_x', 'history_force_z', 'Sthat', 'St',
		'R']
OUT_FILE = '../../data/linear_wave/forces_numerics.csv'

def main():
	"""
	Simulate negatively buoyant particles transported through a linear wave.

	Simulations are performed for particles of varying sizes (Stokes numbers)
	and buoyancies (density ratios), and history effects are included. Results
	are saved to the `data/linear_wave` directory.
	"""
	# create dictionary to store results, initialize variables for simulations
	warnings.filterwarnings('ignore')
	results = {key: [] for key in KEYS}

	# run simulations iteratively
	for st_hat, r in product(STOKES_HATS, RS):
		sols = run_simulation(st_hat, r)
		results = update_results(results, sols[:-3], sols[-3:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def run_simulation(stokes_hat, r):
	"""
	Run a numerical simulation with the specified $\widehat{St}$ and *R*.

	Parameters
	----------
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	r : float
		The ratio between the particle and fluid densities.

	Returns
	-------
	list
		List containing `ndarray` elements of `float` data.
	"""
	# initialize objects
	particle = prt.Particle(stokes_hat)
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, r)

	# initialize local variables
	n = -3
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, wave.period * NUM_PERIODS, DELTA_T)

	# run simulations and return results
	x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, mass_z, \
	   drag_x, drag_z, history_x, history_z = system.maxey_riley(t, y,
											  INCLUDE_HISTORY, HIDE_PROGRESS)
	sols = [t, x, z, xdot, zdot, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x,
			mass_z, drag_x, drag_z, history_x, history_z]
	sols = [sol[:n] for sol in sols]
	sols += [stokes_hat, system.stokes_num, r]
	return sols

if __name__ == '__main__':
	main()
