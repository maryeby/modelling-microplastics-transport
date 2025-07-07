import warnings
import numpy as np
import pandas as pd
from parallelbar import progress_starmap
from itertools import product

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# wave conditions
AMPLITUDE = 0.02
WAVELENGTH = 1.5
DEPTHS = [10, 1.5, 0.5]

# particle conditions
STOKES_NUMS = [0.01, 0.1, 1, 10]
NUM_POINTS = 4  # number of different initial vertical particle positions (z_0s)
X_0 = 0			# initial horizontal particle position

# simulation conditions
NUM_TASKS = len(DEPTHS) * len(STOKES_NUMS) * NUM_POINTS * 2
R = 2 / 3		# denisty ratio
DELTA_T = 1e-3  # timestep
NUM_PERIODS = 3
NUM_CPUS = None
TIMEOUT = 600
HIDE_PROGRESS = True
OUT_FILE = '../../../data/water_wave/st_neutral_numerics.csv'

def main():
	"""
	Run numerical simulations for neutrally buoyant particles in a wave.

	Simulations are run with and without history effects, with various Stokes
	numbers, and in linear waves of arbitrarily deep water. The period-averaged
	Stokes drift velocity is also computed. Results are saved to the
	`data/water_wave` directory.

	See Also
	--------
	models.my_system.compute_drift_velocity
	"""
	waves, repeated_stokes_nums, repeated_z0s = [], [], []
	history = [True] * (NUM_TASKS // 2) + [False] * (NUM_TASKS // 2)
	results = {'z_bar/h': [], 'u_bar': [], 'St': [], 'history': []}
	warnings.filterwarnings('ignore')

	for depth in DEPTHS:
		# create Wave object and compute initial vertical particle positions
		wave = fl.WaterWave(depth, AMPLITUDE, WAVELENGTH)
		z_0s = np.linspace(0, -wave.wavenum * depth, NUM_POINTS, endpoint=False)

		# store St values, z_0 values, and Wave objects for parallel processing
		st, z = zip(*product(STOKES_NUMS, z_0s))
		repeated_stokes_nums += list(st)
		repeated_z0s += list(z)
		waves += [wave] * (len(STOKES_NUMS) * NUM_POINTS)

	# run simulations in parallel 
	params = zip(waves, repeated_stokes_nums, repeated_z0s, history)
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=NUM_TASKS, process_timeout=TIMEOUT)
	for sol in sols: results = update_results(results, [], sol)
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(wave, stokes_num, z_0, include_history):
	"""
	Run a numerical simulation and compute the Stokes drift velocity.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	z_0 : float
		The initial position of the particle.
	include_history : bool
		Whether to include history effects.
	
	Returns
	-------
	list
		A list containing the normalized average vertical particle position,
		the averaged horizontal Stokes drift velocity, the Stokes number, and
		whether history effects were included.
	"""
	# create time series, set the initial position and velocity of the particle
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
	y = [X_0, z_0, xdot_0, zdot_0]

	# initialize the particle and transport system
	particle = prt.Particle(stokes_num)
	system = ts.MyTransportSystem(particle, wave, R)

	# run simulation
	x, z, xdot, _, t, _, _, _, _, _, _, _, _, _, \
	   _ = system.maxey_riley(t, y, include_history, HIDE_PROGRESS)

	# compute averaged horizontal drift velocity and scale results
	_, z_crossings, u, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
	u_bar = np.mean(u) / wave.froude_num
	normalized_z_bar = np.mean(z_crossings) / (wave.wavenum * wave.depth)
	return [normalized_z_bar, u_bar, stokes_num, include_history]

if __name__ == '__main__':
	main()
