import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import repeat, product
from parallelbar import progress_starmap
from tqdm import tqdm

from utils.data_tools import NEUTRAL_R, update_results
from transport_framework import particle as prt
from models import stokes_wave as fl
from models import my_system as ts
from examples.linear_wave.r_analysis.numerics import STOKES_HAT, X_0, DEPTH, \
	 WAVELENGTH, RS, NUM_PERIODS, DELTA_T

AMPLITUDES = [0.01, 0.03, 0.05, 0.07]
NUM_CPUS = None
TIMEOUT = None
HIDE_PROGRESS = True
KEYS = ['z_bar', 'u_d_bar', 'R', 'z_0', 'history', 'A\'', 'epsilon',
		'mean_speed']
OUT_FILE = '../../data/stokes_wave/r_numerics.csv'

def main():
	"""
	Run numerical simulations for particles of varying buoyancy in a wave.

	Simulations are performed for particles of varying buoyancy in fifth order
	Stokes waves of deep water, with and without history effects. Results are
	saved to the `data/stokes_wave` directory.
	"""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_HAT)
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	# initialize parameters for iterative simulations
	z_0s = [-1.5, -2, 0, 0]
	repeated_rs = RS[1:]
	history = [True] * len(z_0s)
	params = zip(repeat(particle), repeat(AMPLITUDES[1]), z_0s, repeated_rs,
				 history)

	# run simulations with history effects iteratively
	print('Running simulations with history effects...')
	for i in tqdm(params, total=len(history)):
		sol = run_numerics(*i)
		results = update_results(results, sol[:2], sol[2:])

	# initialize parameters for parallel processing
	z_0s = np.round(np.linspace(-0.02, -7, 10, endpoint=False), 5).tolist()
	z_0s, repeated_amplitudes = zip(*product(z_0s, AMPLITUDES))
	repeated_rs = [RS[0]] * len(z_0s)
	z_0s = list(z_0s) + [-1.5, -2, 0, 0]
	repeated_amplitudes = list(repeated_amplitudes) + ([AMPLITUDES[1]] \
						* (len(RS) - 1))
	repeated_rs += RS[1:]
	history = [False] * len(z_0s)

	# run simulations without history effects in parallel
	print('Running simulations without history effects...')
	params = zip(repeat(particle), repeated_amplitudes, z_0s, repeated_rs,
				 history)
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=len(z_0s), process_timeout=TIMEOUT)
	# store solutions
	for sol in sols:
		if sol[2] == NEUTRAL_R: # if r is neutrally buoyant
			sol[0] = np.mean(sol[0])
			sol[1] = np.mean(sol[1])
			results = update_results(results, [], sol)
		else:
			results = update_results(results, sol[:2], sol[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False, header=False, mode='a')

def run_numerics(particle, a, z_0, r, include_history):
	"""
	Run a numerical simulation and compute the horizontal drift velocity.

	Parameters
	----------
	particle : Particle (obj)
		The particle transported through the wave.
	a : float
		The amplitude of the wave, *A'*.
	z_0 : float
		The initial vertical particle position.
	r : float
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
	# initialize the Wave and TransportSystem objects
	wave = fl.StokesWave(DEPTH, a, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, r)

	# set the initial particle position and velocity
	xdot_0, zdot_0 = system.flow.velocity(X_0, z_0, t=0)
	y = [X_0, z_0, xdot_0, zdot_0]

	# create time series data, shorten the neutrally buoyant simulation time
	n = 3 if r == RS[0] else NUM_PERIODS
	t = np.arange(0, wave.period * n, DELTA_T)

	# run the simulation and compute the drift velocity
	x, z, xdot, zdot, t = system.maxey_riley(t, y, include_history,
											 HIDE_PROGRESS)[:5]
	_, z_crossings, u, _, _ = ts.compute_drift_velocity(x, z, xdot, t)

	# use an alternate method to compute the drift velocity if the first fails
	if z_crossings.any():
		z_crossings = z_crossings[1:]
	else:
		_, z_crossings, u, _, _ = ts.compute_alternate_drift_velocity(x, z,
									 xdot, zdot, t, 3)
	u /= wave.steepness * wave.steepness # scale the drift velocity
	return [z_crossings, u, r, z_0, include_history, a, wave.steepness,
			wave.mean_speed]

if __name__ == '__main__':
	main()
