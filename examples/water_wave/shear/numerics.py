import warnings
import numpy as np
import pandas as pd
from itertools import repeat
from parallelbar import progress_starmap

from utils.data_tools import extract_data, update_results
from transport_framework import particle as prt 
from models import water_wave as fl
from models import my_system as ts

# particle conditions
STOKES_NUMS = np.round(np.linspace(0.01, 0.05, 9).tolist() \
			+ np.arange(0.1, 1.1, 0.1).tolist(), 3)
ST_TO_SHOW = 0.01	# St to use when varying beta
NUM_POINTS = 12

# create a circle of initial particle positions
RADIUS = 0.1
ANGLE = np.linspace(0, 2 * np.pi, NUM_POINTS, endpoint=False)
X_CENTER = 2 * RADIUS
Z_NEGATIVE = -2
Z_POSITIVE = -100
X_0S = np.round(RADIUS * np.cos(ANGLE) + X_CENTER, 3)

# wave conditions
DEPTH = 20
AMPLITUDE = 0.02
WAVELENGTH = 1

# density ratio
SCALE = 2 / 3
BETAS = np.array(np.round(np.arange(0.75, 0.96, 0.01), 3).tolist() \
	  + np.round(np.arange(0.955, 0.985, 0.005), 3).tolist() \
	  + np.round(np.arange(0.99, 0.9975, 0.0025), 4).tolist() \
	  + np.round(np.arange(1, 1.21, 0.01), 3).tolist())
BETA_TO_SHOW = 0.905	# beta to use when varying St

# simulation conditions
NUM_TASKS = (len(STOKES_NUMS) + len(BETAS)) * NUM_POINTS * 2
DELTA_T = 5e-3
NUM_CPUS = None
TIMEOUT = 240
HIDE_PROGRESS = True
OUT_FILE = '../../data/water_wave/orbit_shear_numerics.csv'

def main():
	"""
	Simulate particles in a wave and periodically record their positions.

	The particles simulated are transported through a linear wave of arbitrarily
	deep water. The initial positions of the particles vary, as does their
	density and size. Conditions for the simulations were chosen in accordance
	with [1] Figure 5. Results are saved to the `data/water_wave` directory.

	References
	----------
	[^1]: [M. H. DiBenedetto et al. (2022).](
		  https://doi.org/10.1017/jfm.2022.95) Enhanced settling and dispersion
		  of inertial particles in surface waves. *Journal of Fluid Mechanics*
		  936, A38.
	"""
	# create dict to store sols and initialize Wave object
	keys = ['x', 'z', 'x_0', 'z_0', 'St', 'beta', 'history']
	results = {key: [] for key in keys}
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	warnings.filterwarnings('ignore')

	# parameters for parallel processing
	repeated_st = STOKES_NUMS.tolist() + [ST_TO_SHOW] * len(BETAS)
	repeated_beta = [BETA_TO_SHOW] * len(STOKES_NUMS) + BETAS.tolist()
	repeated_history = [False] * len(repeated_st) + [True] * len(repeated_st)
	repeated_st += repeated_st
	repeated_beta += repeated_beta
	point_index = []
	for i in range(NUM_POINTS):
		point_index += [i] * (len(STOKES_NUMS) + len(BETAS)) * 2
	repeated_st *= NUM_POINTS
	repeated_beta *= NUM_POINTS
	repeated_history *= NUM_POINTS

	# run simulations in parallel
	params = zip(repeated_st, repeated_beta, repeat(wave), point_index,
				 repeated_history)
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=NUM_TASKS, process_timeout=TIMEOUT)

	# find indices of data points to save
	period = int(wave.period / DELTA_T)
	indices = []
	for i in range(0, 15, 3): indices.append(i * period)

	# store solutions
	for sol in sols:
		indices = [i for i in indices if i < len(sol[0])]
		sol[0] = sol[0][indices]
		sol[1] = sol[1][indices]
		results = update_results(results, sol[:2], sol[2:])
		indices = []
		for i in range(0, 15, 3): indices.append(i * period)
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(stokes_num, beta, wave, i, include_history):
	"""
	Run a numerical simulation with the specified initial position.

	Parameters
	----------
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	beta : float
		The ratio between the particle and fluid densities.
	wave : Wave (obj)
		The wave through which the particle is transported.
	i : int
		The index of the initial particle position elements.
	include_history : bool
		Whether to include history effects.

	Returns
	-------
	list
		A list containing the particle positions at each period, the initial
		position, and whether history effects were included.
	"""
	# initialize time series, create Particle and TransportSystem objects
	num_periods = 16 if beta < 1.15 else 7
	t = np.arange(0, num_periods * wave.period, DELTA_T)
	particle = prt.Particle(stokes_num)
	system = ts.MyTransportSystem(particle, wave, SCALE * beta)

	# set initial particle position and velocity
	z_center = Z_NEGATIVE * RADIUS if beta < 1 else Z_POSITIVE * RADIUS
	z_0s = np.round(RADIUS * np.sin(ANGLE) + z_center, 3)
	xdot_0, zdot_0 = system.flow.velocity(X_0S[i], z_0s[i], t=0)
	y = [X_0S[i], z_0s[i], xdot_0, zdot_0]

	# run simulation
	x, z = system.maxey_riley(t, y, include_history, HIDE_PROGRESS)[:2]
	return [x, z, X_0S[i], z_0s[i], stokes_num, beta, include_history]

if __name__ == '__main__':
	main()
