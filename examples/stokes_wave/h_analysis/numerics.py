import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import repeat, product
from parallelbar import progress_starmap

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import stokes_wave as fl
from models import my_system as ts
from examples.linear_wave.st_analysis.numerics import X_0, STOKES_HATS, \
	 WAVELENGTH
from examples.linear_wave.st_analysis.numerics import RS as CONSTANT_RS
from examples.linear_wave.r_analysis.numerics import RS
from examples.linear_wave.r_analysis.numerics import STOKES_HAT as \
													 CONSTANT_ST_HAT

DEPTHS = [0.549106, 0.354646] # Fr = 0.99, 0.95
AMPLITUDE = 0.07
DELTA_T = 1e-2
NUM_CPUS = 4
TIMEOUT = 6000
HIDE_PROGRESS = True
KEYS = ['z_bar', 'u_d_bar', 'z_0', 'mean_speed', 'Sthat', 'St', 'Sthat/gamma',
		'R', 'depth', 'Fr', 'history']
OUT_FILE = '../../data/stokes_wave/h_numerics.csv'

def main():
	"""
	Run numerical simulations for inertial particles in a 5th order Stokes wave.

	Simulations are run with and without history effects, with positive and
	negative buoyancy, with various Stokes numbers, various depths, and in fifth
	order Stokes waves of deep water. The period-averaged Stokes drift velocity
	is also computed. Results are saved to the `data/stokes_wave` directory.

	See Also
	--------
	models.my_system.compute_drift_velocity()
	"""
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	# create lists of parameters to pass to the run_numerics function
	repeated_st = [STOKES_HATS[0]] + STOKES_HATS[2:4] + STOKES_HATS[:3] \
				+ [CONSTANT_ST_HAT] * len(RS[1:-1])
	repeated_rs = [CONSTANT_RS[0]] * 3 + [CONSTANT_RS[1]] * 3 + RS[1:-1]
	repeated_rs, _ = zip(*product(repeated_rs, DEPTHS))
	repeated_st, repeated_hs = zip(*product(repeated_st, DEPTHS))

	# run simulations with history effects iteratively
	print('Running simulations with history effects...')
	repeated_history = [True] * len(repeated_st)
	params = zip(repeated_st, repeated_rs, repeated_hs, repeated_history)
	for st, r, h, history in tqdm(params, total=len(repeated_st)):
		sol = run_numerics(st, r, h, history)
		results = update_results(results, sol[:2], sol[2:])

	# run simulations without history effects in parallel
	print('Running simulations without history effects...')
	repeated_history = [False] * len(repeated_st)
	params = zip(repeated_st, repeated_rs, repeated_hs, repeated_history)
	sols = progress_starmap(run_numerics, params, process_timeout=TIMEOUT,
							n_cpu=NUM_CPUS, total=len(repeated_st))
	# store results
	for sol in sols: results = update_results(results, sol[:2], sol[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(stokes_hat, r, h, include_history):
	"""
	Run a numerical simulation and compute the horizontal drift velocity.

	Parameters
	----------
	stokes_hat : float
		The Stokes number to use for initializing the particle.
	r : float
		The ratio between the particle and fluid densities.
	h : float
		The depth *h'* of the water.
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
	wave = fl.StokesWave(h, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, r)
	num_periods = 10 if (r == RS[2] or r == RS[-1]
									or stokes_hat == STOKES_HATS[2]
									or stokes_hat == STOKES_HATS[-1]) else 25
	t = np.arange(0, wave.period * num_periods, DELTA_T)

	# set initial position and velocity of the particle
	z_0 = -0.75 if r < 2 / 3 else -wave.wavenum * h * 0.9
	z_0 = -wave.wavenum * h * 0.75 if r == CONSTANT_RS[1] else z_0
	xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
	y = [X_0, z_0, xdot_0, zdot_0]

	# run simulation and compute drift velocity
	x, z, xdot, zdot, t = system.maxey_riley(t, y, include_history,
											 HIDE_PROGRESS)[:5]
	_, z_cross, u, _, _ = ts.compute_drift_velocity(x, z, xdot, t)

	# use an alternate method to compute the drift velocity if the first failed
	if 3 < len(z_cross):
		z_cross = z_cross[1:]
	else:
		_, z_cross, u, _, _ = ts.compute_alternate_drift_velocity(x, z, xdot,
								 zdot, t, num_periods)
	u /= wave.steepness * wave.steepness # scale the drift velocity
	return [z_cross, u, z_0, wave.mean_speed, stokes_hat, system.stokes_num,
			np.round(stokes_hat / system.gamma, 5), r, h, wave.froude_num,
			include_history]

if __name__ == '__main__':
	main()
