import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm.contrib.itertools import product as tqdm
from itertools import product
from parallelbar import progress_starmap

from utils.data_tools import update_results, extract_data
from utils.colors import print_failure
from transport_framework import particle as prt
from models import stokes_wave as fl
from models import my_system as ts
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 DEPTH, WAVELENGTH
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 STOKES_HAT as DB_ST
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 R as DB_R
from examples.linear_wave.horizontal_displacement.numerics import X_0S, \
	 STOKES_HATS, RS, HIDE_PROGRESS

AMPLITUDE = 0.08
TIMEOUT = None
NUM_CPUS = None
NUM_PERIODS = 15
DELTA_T = 5e-3
KEYS = ['x', 'z', 'x_0', 'z_f', 'Sthat', 'St', 'R', 'history']
OUT_FILE = '../../data/stokes_wave/displacement_numerics.csv'

def main():
	"""
	Run numerical simulations for particles of varying densities and sizes.

	Simulations are run with and without history effects for particles of
	varying sizes (Stokes numbers) and densities as they are transported through
	a linear wave of arbitrarily deep water.
	"""
	# create dict to store solutions, ignore warnings
	results = {key: [] for key in KEYS}
	z_finals = {'z_f': [], 'R': []}
	warnings.filterwarnings('ignore')

	# create parameters to run simulations without history effects in parallel
	repeated_x0 = [i[0] for i in list(product(X_0S, RS))] + [i[0] for i in
				   list(product(X_0S, STOKES_HATS))]
	repeated_st = [i[1] for i in list(product(X_0S, STOKES_HATS))] + [DB_ST] \
				* (len(X_0S) * len(RS))
	repeated_rs = [DB_R] * (len(X_0S) * len(STOKES_HATS)) + [i[1] for i in
				   list(product(X_0S, RS))]
	params = zip(repeated_x0, repeated_st, repeated_rs, repeat(False))

	# run simulations without history effects in parallel
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=len(repeated_x0), process_timeout=TIMEOUT)
	for sol in sols:
		z_finals = update_results(z_finals, [], [sol[1][-1], sol[-2]])
		results = update_results(results, sol[:2], sol[2:])

	# run varying density ratio simulations with history effects
	for r, x_0 in tqdm(RS, X_0S):
		sols = run_numerics(x_0, DB_ST, r, True)
		z_finals = update_results(z_finals, [], [sols[1][-1], sols[-2]])
		results = update_results(results, sols[:2], sols[2:])

	# run varying Stokes number simulations with history effects
	for stokes_hat, x_0 in tqdm(STOKES_HATS, X_0S):
		sols = run_numerics(x_0, stokes_hat, DB_R, True)
		z_finals = update_results(z_finals, [], [sols[1][-1], sols[-2]])
		results = update_results(results, sols[:2], sols[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def run_numerics(x_0, stokes_hat, r, include_history):
	r""" 
	Run a numerical simulation with the specified $\widehat{St}$ and $R$.

	Parameters
	----------
	x_0 : float
		The initial horizontal position of the particle.
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	r : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether history effects are included.

	Returns
	-------
	list
		A list containing the particle positions, $\widehat{St}$, $St$, and $R$.

	Notes
	-----
	A failure message is printed and the trajectory is plotted if the particle
	does not reach the seabed.
	"""
	# create objects, time series data, set initial particle position & vel
	particle = prt.Particle(stokes_hat)
	wave = fl.StokesWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, r)
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	z_0 = -1 if r < 2 / 3 else -2
	xdot_0, zdot_0 = wave.velocity(x_0, z_0, t=0)
	y = [x_0, z_0, xdot_0, zdot_0]

	# run simulation and check if the particle reached the seabed
	x, z = system.maxey_riley(t, y, include_history, HIDE_PROGRESS)[:2]
	return [x, z, x_0, z[-1], stokes_hat, system.stokes_num, r, include_history]

if __name__ == '__main__': main()
