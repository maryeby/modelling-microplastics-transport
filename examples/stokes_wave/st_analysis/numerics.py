import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import repeat
from parallelbar import progress_starmap

from utils.data_tools import update_results, print_parameter
from transport_framework import particle as prt
from models import stokes_wave as fl
from models import my_system as ts
from examples.linear_wave.st_analysis.numerics import X_0, STOKES_HATS, DEPTH,\
	 WAVELENGTH, RS

AMPLITUDE = 0.07
DELTA_T = 1e-2
NUM_CPUS = None
TIMEOUT = 4800
HIDE_PROGRESS = True
KEYS = ['t', 'x', 'z', 'u_bar', 'w_bar', 'Sthat', 'St', 'Sthat/gamma', 'R',
		'history']
OUT_FILE = '../../data/stokes_wave/st_numerics.csv'

def main():
	"""
	Run numerical simulations for inertial particles in a 5th order Stokes wave.

	Simulations are run with and without history effects, with positive and
	negative buoyancy, with various Stokes numbers, and in fifth order Stokes
	waves of deep water. The period-averaged Stokes drift velocity is also
	computed. Results are saved to the `data/stokes_wave` directory.

	See Also
	--------
	models.my_system.compute_drift_velocity
	"""
	# create dict to store sols, initialize variables for the simulations
	results = {key: [] for key in KEYS}
	wave = fl.StokesWave(DEPTH, AMPLITUDE, WAVELENGTH)
	print_parameter('epsilon', wave.steepness)
	print_parameter('Fr', wave.froude_num)
	print_parameter('R', RS[0])
	print_parameter('R', RS[1])
	warnings.filterwarnings('ignore')

	# create parameters to run simulations
	repeated_st = [STOKES_HATS[0]] + STOKES_HATS[-2:] + STOKES_HATS[:3]
	repeated_rs = [RS[0]] * 3 + [RS[1]] * 3

	# run simulations iteratively
	print('Running simulations with history effects...')
	repeated_history = [True] * len(repeated_st)
	for st, r, history in tqdm(zip(repeated_st, repeated_rs, repeated_history),
							   total=len(repeated_st)):
		sol = run_numerics(wave, st, r, history)
		results = update_results(results, sol[:3], [None, None] + sol[8:])
		results = update_results(results, sol[3:8], sol[8:])

	# run simulations in parallel
	print('Running simulations without history effects...')
	repeated_history = [False] * len(repeated_st)
	params = zip(repeat(wave), repeated_st, repeated_rs, repeated_history)
	sols = progress_starmap(run_numerics, params, process_timeout=TIMEOUT,
							n_cpu=NUM_CPUS, total=len(repeated_st))
	# store results
	for sol in sols:
		results = update_results(results, sol[:3], [None, None] + sol[8:])
		results = update_results(results, sol[3:8], sol[8:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(wave, stokes_hat, r, include_history):
	"""
	Run a numerical simulation and compute the horizontal drift velocity.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	stokes_hat : float
		The Stokes number to use for initializing the particle.
	r : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether to include history effects.

	Returns
	-------
	list
		A list containing the vertical particle positions at each period, the
		horizontal drift velocity, the Stokes number, the density ratio, and
		whether history effects were included.
	"""
	# create Particle and TransportSystem objects, time series data
	particle = prt.Particle(stokes_hat)
	system = ts.MyTransportSystem(particle, wave, r)
	num_periods = 25 if stokes_hat == STOKES_HATS[0] else 20
	t = np.arange(0, wave.period * num_periods, DELTA_T)

	# set initial position and velocity of the particle
	if r < 2 / 3:
		z_0 = -0.5 if stokes_hat == STOKES_HATS[0] else 0
	else:
		z_0 = -1 if stokes_hat == STOKES_HATS[0] else -2
	xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
	y = [X_0, z_0, xdot_0, zdot_0]

	# run simulation and compute drift velocity
	x, z, xdot, _, t = system.maxey_riley(t, y, include_history,
										  HIDE_PROGRESS)[:5]
	x_cross, z_cross, u, w, t_cross = ts.compute_drift_velocity(x, z, xdot, t)
	u /= wave.steepness * wave.steepness
	w /= wave.steepness * wave.steepness
	return[t, x, z, t_cross[1:], x_cross[1:], z_cross[1:], u, w, stokes_hat,
		   system.stokes_num, np.round(stokes_hat / system.gamma, 5), r,
		   include_history]

if __name__ == '__main__':
	main()
