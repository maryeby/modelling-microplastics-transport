import warnings
import numpy as np
import pandas as pd
from itertools import product, repeat
from parallelbar import progress_starmap
from tqdm import tqdm

from transport_framework import particle as prt	# Particle class
from models import bichromatic_wave as fl		# Flow (Wave) class
from models import my_system as ts				# TransportSystem class
from utils.data_tools import update_results, NEUTRAL_R

# wave conditions
DEPTH = 2
AMPLITUDES = np.array([0.3, 0.2])
WAVELENGTHS = np.array([36, 50])
SLOPE = -0.0095

# simulation conditions
RS = [0.65, 0.66, 0.66, 0.66, NEUTRAL_R, 0.67]		# density ratio
DELTA_T = 5e-3							# timestep size (recommended <= 5e-3)
NUM_CPUS = None
TIMEOUT = 4800
HIDE_PROGRESS = True

# particle conditions
X_0 = 0									# initial horizontal particle position
GAMMA = 1 / 0.66 - 0.5
STOKES_HATS = np.round(np.array([0.15, 0.15 * GAMMA, 0.25 * GAMMA, 0.5 * GAMMA,
								 0.15, 0.15]), 5).tolist()

# constants for writing the data
KEYS = ['t', 'x', 'z', 'xdot', 'zdot', 'seabed_x', 'seabed_z', 'Sthat', 'S',
		'R', 'slope', 'history']
OUT_FILE = '../../data/bichromatic_wave/trajectory_numerics.csv'

def main():
	"""Run simulations of particles transported through bichromatic waves."""
	# create the Particle, Flow, and TransportSystem objects
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	# initialize variables for parallel processing
	sthats, _ = zip(*product(STOKES_HATS, [0, SLOPE]))
	densities, slopes = zip(*product(RS, [0, SLOPE]))
	params = zip(sthats, slopes, densities, repeat(True))

	print('Running simulations with history effects...')
	for sthat, m, r, history in tqdm(params, total=len(sthats)):
		sol = run_simulations(sthat, m, r, history)
		if sol: results = update_results(results, sol[:7], sol[7:])

	# run simulations in parallel
	print('Running simulations without history effects...')
	params = zip(sthats, slopes, densities, repeat(False))
	sols = progress_starmap(run_simulations, params, process_timeout=TIMEOUT,
							n_cpu=NUM_CPUS, total=len(sthats))
	for sol in sols:
		if sol: results = update_results(results, sol[:7], sol[7:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False, mode='a', header=False)

def run_simulations(sthat, m, r, history):
	"""Run a numerical simulation for a particle in a bichromatic wave."""
	#TODO: reset to run all sims
	if r != NEUTRAL_R: return []
	particle = prt.Particle(sthat)
	wave = fl.BichromaticWave(DEPTH, AMPLITUDES, WAVELENGTHS, m)
	system = ts.MyTransportSystem(particle, wave, r)
	num_periods = 3 if r == RS[0] or r == RS[-1] or sthat == STOKES_HATS[2] \
								  or sthat == STOKES_HATS[3] else 11
	z_0 = -0.32 if NEUTRAL_R < r else 0
	if r == NEUTRAL_R: z_0 = -0.001

	# initialize time series and set initial particle velocity
	t = np.arange(0, num_periods * wave.period[0], DELTA_T)
	xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
	y = [X_0, z_0, xdot_0, zdot_0]

	# run simulation and record seabed datapoints
	x, z, xdot, zdot, t = system.maxey_riley(t, y, history, HIDE_PROGRESS)[:5]
	xbed = np.linspace(np.min(x) - 0.1, np.max(x) + 0.1, len(t))
	zbed = -wave.wavenum[0] * wave.seabed(xbed)
	return [t, x, z, xdot, zdot, xbed, zbed, sthat, sthat / system.gamma, r,
			wave.slope, history]

if __name__ == '__main__': main()
