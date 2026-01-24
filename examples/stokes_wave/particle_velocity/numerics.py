import warnings
import numpy as np
import pandas as pd
from parallelbar import progress_starmap
from scipy.optimize import curve_fit
from scipy.signal import hilbert, find_peaks
from scipy.stats import linregress
from itertools import product
from tqdm.contrib.itertools import product as tqdm

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import stokes_wave as fl
from models import my_system as ts
from examples.linear_wave.particle_velocity.numerics import DEPTH, WAVELENGTH, \
	 NUM_PERIODS, DELTA_T, X_0, Z_0

# variables to determine amplitudes for history & non-history simulations
NON_HISTORY_A = np.round(np.linspace(0.0095, 0.07, 20), 5)
HISTORY_A = NON_HISTORY_A[::4].tolist() + [NON_HISTORY_A[-1]]
AMPLITUDES = NON_HISTORY_A.tolist() + HISTORY_A

# other simulation conditions
STOKES_HATS = np.round([73 / 36, 335 / 33], 5)
RS = [0.54, 0.66]
INCLUDE_HISTORY = [False] * len(NON_HISTORY_A) + [True] * len(HISTORY_A)

# constants for parallel processing
TIMEOUT = 1000
NUM_CPUS = 1
NUM_TASKS = len(STOKES_HATS) * len(AMPLITUDES)
HIDE_PROGRESS = True

# constants for writing the data
KEYS = ['t', 'xdot', 'S', 'Sthat', 'R', 'A\'', 'epsilon', 'history']
OUT_FILE = '../../data/stokes_wave/particle_velocity_numerics.csv'

def main():
	"""Simulate negatively buoyant particles in 5th order Stokes waves."""
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	# run numerics without history effects in parallel
	sthats, amplitudes = zip(*product(STOKES_HATS, AMPLITUDES))
	rs, history = zip(*product(RS, INCLUDE_HISTORY))
	params = zip(sthats, rs, amplitudes, history)
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=NUM_TASKS, process_timeout=TIMEOUT)
	for sol in sols: results = update_results(results, sol[:2], sol[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, mode='a', index=False)

def run_numerics(stokes_hat, r, amplitude, history):
	"""Return numerical solutions from the simulation."""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(stokes_hat)
	wave = fl.StokesWave(DEPTH, amplitude, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, r)

	# initialize time series and set initial particle velocity
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]  # initial particle position and velocity
	
	# run simulation
	xdot, _, t = system.maxey_riley(t, y, history, HIDE_PROGRESS)[2:5]
	return [t, xdot, np.round(stokes_hat / system.gamma, 5), stokes_hat, r,
			amplitude, np.round(wave.steepness, 5), history]

if __name__ == '__main__': main()
