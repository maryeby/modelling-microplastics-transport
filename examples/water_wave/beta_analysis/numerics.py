import warnings
import numpy as np
import pandas as pd
from itertools import repeat, product
from parallelbar import progress_starmap
from tqdm import tqdm

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# particle conditions
STOKES_NUM = 0.1
X_0 = 0

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1

# simulation conditions
SCALE = 2 / 3
BETAS = [1, 1.04, 1.02, 0.9, 0.8]
NUM_PERIODS = 20
DELTA_T = 5e-3
NUM_CPUS = None
TIMEOUT = 600
HIDE_PROGRESS = True
OUT_FILE = '../../data/water_wave/beta_numerics.csv'

def main():
	"""
	Run numerical simulations for particles of varying buoyancy in a wave.

	Simulations are performed for particles of varying buoyancy in linear waves
	of deep water, with and without history effects. Results are saved to the
	`data/water_wave` directory.
	"""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	results = {'z_crossings': [], 'u_bar': [], 'beta': [], 'history': []}
	t = np.arange(0, wave.period * NUM_PERIODS, DELTA_T)
	warnings.filterwarnings('ignore')

	# initialize lists of z_0 and beta values to be used in parallel processing
	z_0s = np.linspace(-0.25, -7, 10, endpoint=False)
	z_0s = np.insert(z_0s, 0, -0.02)
	repeated_betas = [BETAS[0]] * len(z_0s)
	z_0s = z_0s.tolist() + [-2, -2, 0, 0]
	repeated_betas += BETAS[1:]
	history = [True] * len(z_0s) + [False] * len(z_0s)
	z_0s += z_0s
	repeated_betas += repeated_betas

	# run simulations in parallel
	params = zip(repeat(particle), repeat(wave), repeat(t), z_0s,
				 repeated_betas, history)
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=len(z_0s), process_timeout=TIMEOUT)
	# store solutions
	for sol in sols:
		if sol[2] == 1: # if beta is neutrally buoyant
			sol[0] = np.mean(sol[0])
			sol[1] = np.mean(sol[1])
			results = update_results(results, [], sol)
		else:
			results = update_results(results, sol[:2], sol[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(particle, wave, t, z_0, beta, include_history):
	"""
	Run a numerical simulation and compute the horizontal drift velocity.

	Parameters
	----------
	particle : Particle (obj)
		The particle transported through the wave.
	wave : Wave (obj)
		The wave through which the particle is transported.
	t : ndarray
		1D array containing `float` time series data.
	z_0 : float
		The initial vertical particle position.
	beta : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether to include history effects.

	Returns
	-------
	list
		A list containing the vertical particle positions at each period, the
		horizontal drift velocity, the density ratio, and whether history
		effects were included.
	"""
	density_ratio = SCALE * beta
	system = ts.MyTransportSystem(particle, wave, density_ratio)
	xdot_0, zdot_0 = system.flow.velocity(X_0, z_0, t=0)
	y = [X_0, z_0, xdot_0, zdot_0]
	x, z, xdot, _, t = system.maxey_riley(t, y, include_history,
										  HIDE_PROGRESS)[:5]
	_, z_crossings, u, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
	return [z_crossings[1:], u, beta, include_history]

if __name__ == '__main__':
	main()
