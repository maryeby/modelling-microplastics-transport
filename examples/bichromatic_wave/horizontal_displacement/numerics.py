import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parallelbar import progress_starmap
from itertools import repeat, product
from tqdm.contrib.itertools import product as tqdm

from utils.data_tools import update_results
from utils.colors import print_failure
from transport_framework import particle as prt
from models import bichromatic_wave as fl
from models import my_system as ts
from examples.bichromatic_wave.trajectories.numerics import DEPTH, AMPLITUDES,\
	 WAVELENGTHS, SLOPE
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 STOKES_HAT as DB_ST
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 R as DB_R

X_0S = np.round(np.linspace(0, 2 * np.pi, 5, endpoint=False), 5)
NUM_POINTS = 10
STOKES_HATS = np.round(np.linspace(0.15, 1, NUM_POINTS), 5)
RS = np.round(np.linspace(0.54, 0.8, NUM_POINTS), 5)
DELTA_T = 1e-2
TIMEOUT = None
NUM_CPUS = None
HIDE_PROGRESS = True
KEYS = ['x', 'z', 'x_0', 'Sthat', 'St', 'R', 'slope', 'history']
OUT_FILE = '../../data/bichromatic_wave/displacement_numerics.csv'

def main():
	"""
	Run numerical simulations for particles of varying densities and sizes.

	Simulations are run with and without history effects for particles of
	varying sizes (Stokes numbers) and densities as they are transported through
	a bichromatic wave of shallow water.
	"""
	# create dict to store solutions, ignore warnings
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	# run simulations with history effects
	print('Running simulations with history for varying St...')
	for stokes_hat, x_0, m in tqdm(STOKES_HATS, X_0S, [0, SLOPE]):
		sols = run_numerics(stokes_hat, m, DB_R, x_0, True)
		results = update_results(results, sols[:2], sols[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)
	print('Running simulations with history for varying R...')
	for r, x_0, m in tqdm(RS, X_0S, [0, SLOPE]):
		sols = run_numerics(DB_ST, m, r, x_0, True)
		results = update_results(results, sols[:2], sols[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False, mode='a', header=False)

	# variables for parallel processing
	repeated_x0 = [i[0] for i in list(product(X_0S, RS))] + [i[0] for i in
								 list(product(X_0S, STOKES_HATS))]
	repeated_st = [i[1] for i in list(product(X_0S, STOKES_HATS))] + [DB_ST] \
				* (len(X_0S) * NUM_POINTS)
	repeated_rs = [DB_R] * (len(X_0S) * NUM_POINTS) + [i[1] for i in
								 list(product(X_0S, RS))]
	repeated_slopes = [0] * len(repeated_x0) + [SLOPE] * len(repeated_x0)
	repeated_x0 *= 2
	repeated_st *= 2
	repeated_rs *= 2
	params = zip(repeated_st, repeated_slopes, repeated_rs, repeated_x0,
				 repeat(False))

	# run simulatons without history effects
	print('Running simulations without history effects...')
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=len(repeated_x0), process_timeout=TIMEOUT)
	for sol in sols: results = update_results(results, sol[:2], sol[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False, mode='a', header=False)

def run_numerics(stokes_hat, slope, r, x_0, include_history):
	r""" 
	Run a numerical simulation with the specified $\widehat{St}$ and $R$.

	Parameters
	----------
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	slope: float
		The slope of the seabed.
	r : float
		The ratio between the particle and fluid densities.
	x_0 : float
		The initial horizontal position of the particle.
	include_history : bool
		Whether history effects are included.

	Returns
	-------
	list
		A list containing the particle positions, $\widehat{St}$, $St$, and $R$.
	"""
	particle = prt.Particle(stokes_hat)
	wave = fl.BichromaticWave(DEPTH, AMPLITUDES, WAVELENGTHS, slope)
	system = ts.MyTransportSystem(particle, wave, r)
	z_0 = -0.3 if 2 / 3 < r else -0.15
	num_periods = 2 if r < 0.6 or 0.7 < r or 0.4 < stokes_hat else 11
	if r == DB_R:
		num_periods = 3 if 0.35 <= stokes_hat else 6
	else:
		num_periods = 4 if 0.63 <= r <= 0.7 else 1
	xdot_0, zdot_0 = wave.velocity(x_0, z_0, t=0)
	y = [x_0, z_0, xdot_0, zdot_0]
	t = np.arange(0, num_periods * wave.period[0], DELTA_T)
	x, z = system.maxey_riley(t, y, include_history, HIDE_PROGRESS)[:2]
	return [x, z, x_0, stokes_hat, system.stokes_num, r, slope, include_history]


if __name__ == '__main__': main()
