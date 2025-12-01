import warnings
import numpy as np
import pandas as pd
from itertools import repeat
from parallelbar import progress_starmap
from tqdm import tqdm

from utils.data_tools import extract_data, update_results
from transport_framework import particle as prt 
from models import linear_wave as fl
from models import my_system as ts

# particle conditions
NUM_POINTS = 12
ST_TO_SHOW = np.round(1 / (4 * np.pi), 5) # Sthat to use when varying R
STOKES_HATS = np.round(np.linspace(ST_TO_SHOW, 5 / (4 * np.pi), 9).tolist() \
			+ np.linspace(5 / (2 * np.pi), 25 / np.pi, 10).tolist(), 5)

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
WAVELENGTH = 1.5

# density ratio
RS = np.round(np.array([1 / 2, 38 / 75, 77 / 150, 13 / 25, 79 / 150, 8 / 15,
						27 / 50, 41 / 75, 83 / 150, 14 / 25, 17 / 30, 43 / 75,
						29 / 50, 44 / 75, 89 / 150, 3 / 5, 91 / 150, 46 / 75,
						31 / 50, 47 / 75, 19 / 30, 191 / 300, 16 / 25,
						193 / 300, 97 / 150, 13 / 20, 49 / 75, 197 / 300,
						33 / 50, 397 / 600, 199 / 300, 133 / 200, 2 / 3,
						101 / 150, 17 / 25, 103 / 150, 52 / 75, 7 / 10, 53 / 75,
						107 / 150, 18 / 25, 109 / 150, 11 / 15, 37 / 50,
						56 / 75, 113 / 150, 19 / 25, 23 / 30, 58 / 75, 39 / 50,
						59 / 75, 119 / 150, 4 / 5]), 5)
R_TO_SHOW = np.round(181 / 300, 5)

# simulation conditions
NUM_TASKS = (len(STOKES_HATS) + len(RS)) * NUM_POINTS
DELTA_T = 5e-3
NUM_CPUS = None
TIMEOUT = 240
HIDE_PROGRESS = True
KEYS = ['x', 'z', 'x_0', 'z_0', 'Sthat', 'Sthat/gamma', 'R', 'history']
OUT_FILE = '../../data/linear_wave/orbit_shear_numerics.csv'

def main():
	"""
	Simulate particles in a wave and periodically record their positions.

	The particles simulated are transported through a linear wave of arbitrarily
	deep water. The initial positions of the particles vary, as does their
	density and size. Conditions for the simulations were chosen in accordance
	with [1] Figure 5. Results are saved to the `data/linear_wave` directory.

	References
	----------
	[^1]: [M. H. DiBenedetto et al. (2022).](
		  https://doi.org/10.1017/jfm.2022.95) Enhanced settling and dispersion
		  of inertial particles in surface waves. *Journal of Fluid Mechanics*
		  936, A38.
	"""
	# create dict to store sols and initialize Wave object
	results = {key: [] for key in KEYS}
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	warnings.filterwarnings('ignore')

	# parameters for parallel processing
	repeated_st = STOKES_HATS.tolist() + [ST_TO_SHOW] * len(RS)
	repeated_r = [R_TO_SHOW] * len(STOKES_HATS) + RS.tolist()
	repeated_history = [True] * len(repeated_st)
	point_index = []
	for i in range(NUM_POINTS):
		point_index += [i] * (len(STOKES_HATS) + len(RS)) * 2
	repeated_st *= NUM_POINTS
	repeated_r *= NUM_POINTS
	repeated_history *= NUM_POINTS

	# run simulations iteratively with history effects
	params = zip(repeated_st, repeated_r, repeat(wave), point_index,
				 repeated_history)
	sols1 = []
	for p in tqdm(params, total=NUM_TASKS):
		sol = run_numerics(*p)
		sols1.append(sol)

	# run simulations in parallel without history effects
	repeated_history = [False] * len(repeated_history)
	params = zip(repeated_st, repeated_r, repeat(wave), point_index,
				 repeated_history)
	sols2 = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=NUM_TASKS, process_timeout=TIMEOUT)

	# find indices of data points to save
	period = int(wave.period / DELTA_T)
	indices = []
	for i in range(0, 15, 3): indices.append(i * period)

	# store solutions
	sols = sols1 + sols2
	for sol in sols:
		indices = [i for i in indices if i < len(sol[0])]
		sol[0] = sol[0][indices]
		sol[1] = sol[1][indices]
		results = update_results(results, sol[:2], sol[2:])
		indices = []
		for i in range(0, 15, 3): indices.append(i * period)
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(stokes_hat, r, wave, i, include_history):
	"""
	Run a numerical simulation with the specified initial position.

	Parameters
	----------
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	r : float
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
	num_periods = 16 if r < 0.75 else 7
	t = np.arange(0, num_periods * wave.period, DELTA_T)
	particle = prt.Particle(stokes_hat)
	system = ts.MyTransportSystem(particle, wave, r)

	# set initial particle position and velocity
	z_center = Z_NEGATIVE * RADIUS if r < 1 else Z_POSITIVE * RADIUS
	z_0s = np.round(RADIUS * np.sin(ANGLE) + z_center, 3)
	xdot_0, zdot_0 = system.flow.velocity(X_0S[i], z_0s[i], t=0)
	y = [X_0S[i], z_0s[i], xdot_0, zdot_0]

	# run simulation
	x, z = system.maxey_riley(t, y, include_history, HIDE_PROGRESS)[:2]
	return [x, z, X_0S[i], z_0s[i], stokes_hat, stokes_hat / system.gamma, r,
			include_history]

if __name__ == '__main__':
	main()
